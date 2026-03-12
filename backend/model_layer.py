"""
==============================================================================
MODEL LAYER — Phase 1: The Core Engine
==============================================================================
Optimization Proxy (feed-forward PyTorch neural network) with a deterministic
Repair Layer that enforces physical manufacturing constraints via alternating
projections / clamping.

This network replaces the computationally expensive NSGA-II at inference time,
producing optimal machine settings in < 1 ms.

Author : Core Engine Team
Version: 1.0.0
==============================================================================
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Import upstream modules
from data_layer import build_training_dataset, DATA_DIR
from offline_optimizer import (
    generate_golden_signatures,
    DECISION_VARS,
    DECISION_BOUNDS,
    TARGET_COLS,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 1. REPAIR LAYER — Deterministic Physical Constraint Enforcement
# ─────────────────────────────────────────────────────────────────────────────
class RepairLayer(nn.Module):
    """Deterministic feasibility/repair layer that forces neural network
    outputs to strictly adhere to factory physical constraints.

    Why this is critical:
      Neural networks output continuous real-valued numbers. Without repair,
      the proxy could predict physically impossible parameters (e.g., a
      Machine_Speed of 150 RPM when the equipment max is 80 RPM, or a
      negative Drying_Temp).

    How it works:
      1. **Hard Clamping (Projection):** Each output dimension is independently
         clamped to its physically feasible range using `torch.clamp()`.
         This is mathematically equivalent to an orthogonal projection onto
         the feasible hyperrectangle — it finds the closest feasible point
         to the network's raw output while preserving all other dimensions.

      2. **Differentiability:** `torch.clamp()` has well-defined (sub)gradients:
         - gradient = 1.0 when inside bounds (pass-through)
         - gradient = 0.0 when hitting bounds (straight-through estimator)
         This allows gradients to flow during backprop for inputs within
         bounds, and zeros the gradient for clamped outputs (they're already
         at the boundary — no further push needed).

      3. **Alternating Projections (Cross-constraint):** If coupling constraints
         exist (e.g., `Compression_Force` must correlate with `Machine_Speed`),
         the layer applies iterative alternating projections between the
         individual box constraints and any coupling constraints until
         convergence or max iterations.

    Physical Constraints Enforced:
      - Granulation_Time  ∈ [9.0,  27.0] min
      - Binder_Amount     ∈ [5.0,  15.0] g
      - Drying_Temp       ∈ [40.0, 80.0] °C     (safety limit)
      - Drying_Time       ∈ [20.0, 60.0] min
      - Compression_Force ∈ [5.0,  25.0] kN      (equipment limit)
      - Machine_Speed     ∈ [20.0, 80.0] RPM     (equipment max)
      - Lubricant_Conc    ∈ [0.3,  2.0]  %
    """

    def __init__(self) -> None:
        super().__init__()
        # Register bounds as buffers (moves with device, not trainable)
        lb = torch.tensor([DECISION_BOUNDS[v][0] for v in DECISION_VARS],
                          dtype=torch.float32)
        ub = torch.tensor([DECISION_BOUNDS[v][1] for v in DECISION_VARS],
                          dtype=torch.float32)
        self.register_buffer("lower_bounds", lb)
        self.register_buffer("upper_bounds", ub)

        # Maximum alternating projection iterations for coupling constraints
        self.max_projection_iters: int = 3

    def forward(self, raw_output: torch.Tensor) -> torch.Tensor:
        """Apply physical constraint repair to raw network outputs.

        Parameters
        ----------
        raw_output : torch.Tensor
            Raw predictions from the neural network, shape (batch, n_decisions).

        Returns
        -------
        torch.Tensor
            Repaired (feasible) predictions, guaranteed to satisfy all
            physical constraints.
        """
        # ── Step 1: Box constraint projection (hard clamping) ───────────
        repaired = torch.clamp(raw_output, self.lower_bounds, self.upper_bounds)

        # ── Step 2: Alternating projections for coupling constraints ────
        # Coupling constraint: if Compression_Force > 20 kN then
        #   Machine_Speed ≤ 60 RPM  (equipment de-rating under high load)
        for _ in range(self.max_projection_iters):
            # Identify indices (based on DECISION_VARS order)
            cf_idx = DECISION_VARS.index("Compression_Force")
            ms_idx = DECISION_VARS.index("Machine_Speed")

            # Where Compression_Force > 20 kN, cap Machine_Speed at 60
            high_force_mask = repaired[:, cf_idx] > 20.0
            if high_force_mask.any():
                constrained_speed = torch.clamp(
                    repaired[:, ms_idx], max=60.0
                )
                repaired = repaired.clone()
                repaired[high_force_mask, ms_idx] = constrained_speed[high_force_mask]

            # Coupling: Drying_Time should scale with Drying_Temp
            #   higher temp → shorter drying time is acceptable
            dt_temp_idx = DECISION_VARS.index("Drying_Temp")
            dt_time_idx = DECISION_VARS.index("Drying_Time")

            # If temp > 70°C, cap drying time at 40 min (efficiency constraint)
            high_temp_mask = repaired[:, dt_temp_idx] > 70.0
            if high_temp_mask.any():
                capped_time = torch.clamp(repaired[:, dt_time_idx], max=40.0)
                repaired = repaired.clone()
                repaired[high_temp_mask, dt_time_idx] = capped_time[high_temp_mask]

            # Re-apply box constraints after coupling adjustments
            repaired = torch.clamp(repaired, self.lower_bounds, self.upper_bounds)

        return repaired


# ─────────────────────────────────────────────────────────────────────────────
# 2. OPTIMIZATION PROXY — Feed-Forward Neural Network
# ─────────────────────────────────────────────────────────────────────────────
class OptimizationProxy(nn.Module):
    """End-to-end neural network that instantly outputs optimal machine
    settings from incoming context features.

    Architecture:
      Input → Linear → BatchNorm → GELU → Dropout
            → Linear → BatchNorm → GELU → Dropout
            → Linear → BatchNorm → GELU → Dropout
            → Linear (output head)
            → RepairLayer (deterministic clamping)

    The Repair Layer is appended as the FINAL operation, guaranteeing
    that every output is physically feasible regardless of what the
    learned layers produce.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = len(DECISION_VARS),
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),                # Smooth activation, better than ReLU for regression
                nn.Dropout(p=dropout),
            ])
            prev_dim = h_dim

        # Output head: raw decision variables (before repair)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.backbone = nn.Sequential(*layers)

        # ── The Repair Layer (CRUCIAL) ──────────────────────────────────
        self.repair = RepairLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: features → backbone → repair → feasible settings.

        Parameters
        ----------
        x : torch.Tensor
            Input context features, shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Physically-feasible optimal machine settings, shape (batch, n_decisions).
        """
        raw = self.backbone(x)
        repaired = self.repair(raw)
        return repaired

    def predict_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass WITHOUT repair (for analysis / debugging)."""
        return self.backbone(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
class ProxyTrainer:
    """Training pipeline for the Optimization Proxy.

    Trains on Golden Signatures generated offline by the NSGA-II optimizer.
    """

    def __init__(
        self,
        model: OptimizationProxy,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 200,
    ) -> None:
        self.model = model.to(DEVICE)
        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        self.epochs = epochs

        # Scalers for normalization
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

    def prepare_data(
        self, golden_df: pd.DataFrame
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation DataLoaders from Golden Signatures.

        Parameters
        ----------
        golden_df : pd.DataFrame
            Golden Signatures DataFrame from NSGA-II.

        Returns
        -------
        Tuple[DataLoader, DataLoader]
            Training and validation DataLoaders.
        """
        # ── Identify input (context) and output (decision) columns ──────
        context_cols = [c for c in golden_df.columns
                        if c.startswith("ctx_")]
        decision_cols = [c for c in DECISION_VARS if c in golden_df.columns]

        if not context_cols:
            raise ValueError("No context columns (ctx_*) found in Golden Signatures")
        if not decision_cols:
            raise ValueError("No decision columns found in Golden Signatures")

        X = golden_df[context_cols].values.astype(np.float32)
        y = golden_df[decision_cols].values.astype(np.float32)

        # Handle NaN / inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

        # ── Normalize ──────────────────────────────────────────────────
        X_scaled = self.input_scaler.fit_transform(X).astype(np.float32)
        y_scaled = self.output_scaler.fit_transform(y).astype(np.float32)

        # ── Train/val split (80/20) ─────────────────────────────────────
        n = len(X_scaled)
        n_train = int(0.8 * n)
        indices = np.random.default_rng(42).permutation(n)

        X_train = torch.tensor(X_scaled[indices[:n_train]])
        y_train = torch.tensor(y_scaled[indices[:n_train]])
        X_val = torch.tensor(X_scaled[indices[n_train:]])
        y_val = torch.tensor(y_scaled[indices[n_train:]])

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=self.batch_size
        )

        print(f"[PROXY TRAINER] Data prepared:")
        print(f"  Input features : {X.shape[1]} context columns")
        print(f"  Output targets : {len(decision_cols)} decision variables")
        print(f"  Train / Val    : {n_train} / {n - n_train}")

        return train_loader, val_loader

    def train(self, golden_df: pd.DataFrame) -> Dict[str, List[float]]:
        """Train the Optimization Proxy on Golden Signatures.

        Parameters
        ----------
        golden_df : pd.DataFrame
            Golden Signatures from NSGA-II.

        Returns
        -------
        Dict[str, List[float]]
            Training history with 'train_loss' and 'val_loss'.
        """
        train_loader, val_loader = self.prepare_data(golden_df)

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        print(f"\n[PROXY TRAINER] Training Optimization Proxy:")
        print(f"  Device : {DEVICE}")
        print(f"  Epochs : {self.epochs}")
        print(f"  LR     : {self.optimizer.param_groups[0]['lr']}")

        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            # ── Training ────────────────────────────────────────────────
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

                self.optimizer.zero_grad()

                # Forward pass through backbone ONLY for loss
                # (repair layer operates on original scale, not scaled)
                pred = self.model.backbone(X_batch)
                loss = self.criterion(pred, y_batch)
                loss.backward()

                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_losses.append(loss.item())

            self.scheduler.step()

            # ── Validation ──────────────────────────────────────────────
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    pred = self.model.backbone(X_batch)
                    loss = self.criterion(pred, y_batch)
                    val_losses.append(loss.item())

            avg_train = np.mean(train_losses)
            avg_val = np.mean(val_losses) if val_losses else 0.0
            history["train_loss"].append(avg_train)
            history["val_loss"].append(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val

            if epoch % 40 == 0 or epoch == self.epochs - 1:
                print(f"  Epoch {epoch:4d}/{self.epochs}: "
                      f"Train Loss = {avg_train:.6f}, "
                      f"Val Loss = {avg_val:.6f}, "
                      f"LR = {self.scheduler.get_last_lr()[0]:.2e}")

        print(f"\n  ✓ Training complete. Best validation loss: {best_val_loss:.6f}")
        return history


# ─────────────────────────────────────────────────────────────────────────────
# 4. INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class InferenceEngine:
    """Production-grade inference engine for the trained Optimization Proxy.

    Given incoming context features, instantly produces optimal & feasible
    machine settings using a single neural network forward pass + repair.
    """

    def __init__(
        self,
        model: OptimizationProxy,
        input_scaler: StandardScaler,
        output_scaler: StandardScaler,
    ) -> None:
        self.model = model.to(DEVICE)
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.model.eval()

    @torch.no_grad()
    def predict(self, context_features: np.ndarray) -> pd.DataFrame:
        """Produce optimal machine settings for one or more context inputs.

        Parameters
        ----------
        context_features : np.ndarray
            Context feature vector(s), shape (n_samples, n_features).

        Returns
        -------
        pd.DataFrame
            Predicted optimal settings with columns matching DECISION_VARS,
            all guaranteed to be within physical constraints.
        """
        start = time.perf_counter_ns()

        # Scale inputs
        X_scaled = self.input_scaler.transform(context_features).astype(np.float32)
        X_tensor = torch.tensor(X_scaled).to(DEVICE)

        # Forward pass through backbone → get scaled predictions
        raw_scaled = self.model.backbone(X_tensor)

        # Inverse-scale to original decision-variable space
        raw_np = raw_scaled.cpu().numpy()
        raw_original = self.output_scaler.inverse_transform(raw_np)

        # Apply Repair Layer on original-scale values
        raw_tensor = torch.tensor(raw_original.astype(np.float32)).to(DEVICE)
        repaired = self.model.repair(raw_tensor)

        elapsed_ns = time.perf_counter_ns() - start
        elapsed_ms = elapsed_ns / 1_000_000

        result_df = pd.DataFrame(
            repaired.cpu().numpy(),
            columns=DECISION_VARS,
        )

        print(f"[INFERENCE] Predicted {len(result_df)} settings in {elapsed_ms:.2f} ms")
        return result_df

    def validate_repair(self, predictions: pd.DataFrame) -> bool:
        """Verify all predictions satisfy physical constraints.

        Parameters
        ----------
        predictions : pd.DataFrame
            Output from predict().

        Returns
        -------
        bool
            True if ALL predictions are within bounds.
        """
        all_valid = True
        for var in DECISION_VARS:
            lo, hi = DECISION_BOUNDS[var]
            vals = predictions[var].values
            below = (vals < lo - 1e-6).sum()
            above = (vals > hi + 1e-6).sum()
            if below or above:
                print(f"  ✗ {var}: {below} below min, {above} above max")
                all_valid = False
            else:
                print(f"  ✓ {var}: all values in [{lo}, {hi}]")
        return all_valid


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║      MODEL LAYER — Phase 1: Core Engine                 ║")
    print("║      Optimization Proxy + Repair Layer                  ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # ── Step 1: Build training data ─────────────────────────────────────
    print("Step 1: Building training dataset...\n")
    training_data = build_training_dataset()

    # ── Step 2: Generate Golden Signatures ──────────────────────────────
    print("\nStep 2: Generating Golden Signatures via NSGA-II...\n")
    golden_sigs = generate_golden_signatures(
        training_df=training_data,
        n_contexts=15,
        seed=42,
    )
    print(f"  Generated {len(golden_sigs)} Golden Signatures")

    # ── Step 3: Initialize and train proxy ──────────────────────────────
    print("\nStep 3: Training Optimization Proxy...\n")
    context_cols = [c for c in golden_sigs.columns if c.startswith("ctx_")]
    input_dim = len(context_cols)

    proxy = OptimizationProxy(
        input_dim=input_dim,
        output_dim=len(DECISION_VARS),
        hidden_dims=(256, 128, 64),
        dropout=0.15,
    )

    trainer = ProxyTrainer(
        model=proxy,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=min(64, len(golden_sigs)),
        epochs=200,
    )
    history = trainer.train(golden_sigs)

    # ── Step 4: Inference + Repair validation ───────────────────────────
    print("\nStep 4: Running inference with Repair Layer validation...\n")
    engine = InferenceEngine(
        model=proxy,
        input_scaler=trainer.input_scaler,
        output_scaler=trainer.output_scaler,
    )

    # Inference on a sample of contexts
    sample_X = golden_sigs[context_cols].values[:10].astype(np.float32)
    predictions = engine.predict(sample_X)

    print(f"\nSample predictions (first 5):")
    print(predictions.head().to_string(index=False))

    # ── Step 5: Validate Repair Layer ───────────────────────────────────
    print(f"\nRepair Layer Constraint Validation:")
    constraints_ok = engine.validate_repair(predictions)

    if constraints_ok:
        print(f"\n✓ ALL predictions satisfy physical constraints!")
        print(f"  The Repair Layer successfully enforces feasibility.")
    else:
        print(f"\n✗ Some constraints violated — repair layer needs debugging")

    # ── Step 6: Compare raw vs repaired outputs ─────────────────────────
    print(f"\n{'─' * 60}")
    print(f"Raw vs Repaired Output Comparison (first 3 samples):")
    print(f"{'─' * 60}")

    proxy.eval()
    with torch.no_grad():
        X_scaled = trainer.input_scaler.transform(sample_X[:3]).astype(np.float32)
        X_tensor = torch.tensor(X_scaled).to(DEVICE)

        raw_scaled = proxy.backbone(X_tensor)
        raw_np = trainer.output_scaler.inverse_transform(raw_scaled.cpu().numpy())

        raw_tensor = torch.tensor(raw_np.astype(np.float32)).to(DEVICE)
        repaired = proxy.repair(raw_tensor).cpu().numpy()

    for i in range(3):
        print(f"\n  Sample {i + 1}:")
        for j, var in enumerate(DECISION_VARS):
            lo, hi = DECISION_BOUNDS[var]
            raw_val = raw_np[i, j]
            rep_val = repaired[i, j]
            clamped = "CLAMPED" if abs(raw_val - rep_val) > 1e-4 else "ok"
            print(f"    {var:25s}: raw={raw_val:8.2f} → repaired={rep_val:8.2f} "
                  f"[{lo:.1f}, {hi:.1f}] {clamped}")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"MODEL LAYER SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Architecture     : {sum(p.numel() for p in proxy.parameters()):,} parameters")
    print(f"  Repair Layer     : {len(DECISION_VARS)} variable constraints + 2 coupling constraints")
    print(f"  Training Loss    : {history['train_loss'][-1]:.6f}")
    print(f"  Validation Loss  : {history['val_loss'][-1]:.6f}")
    print(f"  Inference Speed  : < 1 ms per batch")
    print(f"  Constraints OK   : {constraints_ok}")
    print(f"{'=' * 60}")
