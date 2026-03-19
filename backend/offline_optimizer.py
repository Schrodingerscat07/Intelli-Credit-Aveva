"""
==============================================================================
OFFLINE OPTIMIZER — Phase 1: The Core Engine
==============================================================================
Multi-output XGBoost surrogate model + NSGA-II genetic algorithm for offline
Pareto optimization.  Generates "Golden Signatures" — Pareto-optimal machine
settings for training the real-time Optimization Proxy.

Author : Core Engine Team
Version: 1.0.0
==============================================================================
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# Import the data layer
from data_layer import build_training_dataset, DATA_DIR

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Decision variables the optimizer controls
DECISION_VARS: List[str] = [
    "Granulation_Time",
    "Binder_Amount",
    "Drying_Temp",
    "Drying_Time",
    "Compression_Force",
    "Machine_Speed",
    "Lubricant_Conc",
]

# Target outputs the surrogate predicts
TARGET_COLS: List[str] = [
    "Tablet_Weight",
    "Hardness",
    "Friability",
    "Power_Consumption_kW",
]

# Physical bounds for decision variables  (min, max)
DECISION_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Granulation_Time":  (9.0,  27.0),
    "Binder_Amount":     (5.0,  15.0),
    "Drying_Temp":       (40.0, 80.0),
    "Drying_Time":       (20.0, 60.0),
    "Compression_Force": (5.0,  25.0),
    "Machine_Speed":     (20.0, 80.0),
    "Lubricant_Conc":    (0.3,  2.0),
}

# Quality constraints for feasibility
QUALITY_CONSTRAINTS: Dict[str, Tuple[float, float]] = {
    "Friability": (0.1, 1.0),   # must stay within acceptable pharma range
    "Hardness":   (4.0, 10.0),  # kP — too soft or too hard is reject
}

# NSGA-II hyperparameters
NSGA_POP_SIZE: int = 200
NSGA_GENERATIONS: int = 150
NSGA_CROSSOVER_RATE: float = 0.9
NSGA_MUTATION_RATE: float = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# 1. SURROGATE MODEL — Multi-output XGBoost
# ─────────────────────────────────────────────────────────────────────────────
class SurrogateModel:
    """Multi-output XGBoost regressor that maps machine settings + features
    to quality and energy targets.

    Wraps sklearn's MultiOutputRegressor around XGBRegressor for
    independent per-target training with shared input features.
    """

    def __init__(self, n_estimators: int = 300, max_depth: int = 6,
                 learning_rate: float = 0.05, seed: int = 42) -> None:
        self.model = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                verbosity=0,
            )
        )
        self.feature_names: List[str] = []
        self.target_names: List[str] = TARGET_COLS
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "SurrogateModel":
        """Train the surrogate on the merged training dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Merged dataset from data_layer.build_training_dataset().

        Returns
        -------
        SurrogateModel
            Self (for chaining).
        """
        # Identify available targets (Power_Consumption_kW may come from
        # telemetry features with a phase prefix — use a mean aggregation)
        available_targets: List[str] = []
        y_cols: List[str] = []

        for t in self.target_names:
            if t in df.columns:
                available_targets.append(t)
                y_cols.append(t)
            elif t == "Power_Consumption_kW":
                # Synthesize from phase-level Power features
                power_cols = [c for c in df.columns
                              if "Power_Consumption_kW_mean" in c]
                if power_cols:
                    df["Power_Consumption_kW"] = df[power_cols].mean(axis=1)
                    available_targets.append(t)
                    y_cols.append(t)
                    print(f"  ℹ Synthesized 'Power_Consumption_kW' from "
                          f"{len(power_cols)} phase-level power features")

        self.target_names = available_targets
        y = df[y_cols].values

        # Features = everything except targets, Batch_ID, and other non-numeric
        exclude = set(y_cols) | {"Batch_ID"}
        self.feature_names = [c for c in df.columns
                              if c not in exclude
                              and df[c].dtype in [np.float64, np.int64, float, int]]
        X = df[self.feature_names].values

        print(f"[SURROGATE] Training on {X.shape[0]} samples, "
              f"{X.shape[1]} features → {len(y_cols)} targets")

        self.model.fit(X, y)
        self._is_fitted = True

        # Cross-validation report
        print(f"[SURROGATE] Cross-validation R² scores:")
        for i, target in enumerate(y_cols):
            scores = cross_val_score(
                XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                             subsample=0.8, random_state=42, verbosity=0),
                X, y[:, i], cv=min(5, len(X)), scoring="r2"
            )
            print(f"  {target:30s} → R² = {scores.mean():.4f} ± {scores.std():.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets from feature array.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted targets, shape (n_samples, n_targets).
        """
        if not self._is_fitted:
            raise RuntimeError("Surrogate model not fitted yet")
        return self.model.predict(X)

    def predict_from_decisions(
        self, decisions: np.ndarray, context_row: np.ndarray
    ) -> np.ndarray:
        """Predict targets from decision-variable values + static context.

        Parameters
        ----------
        decisions : np.ndarray
            Decision variable values, shape (n_samples, n_decisions).
        context_row : np.ndarray
            Context features (telemetry + non-decision production features),
            shape (n_context,).

        Returns
        -------
        np.ndarray
            Predicted targets.
        """
        n_samples = decisions.shape[0]
        # Build full feature vector by repeating context for each candidate
        full_X = np.zeros((n_samples, len(self.feature_names)))
        for i, fname in enumerate(self.feature_names):
            if fname in DECISION_VARS:
                dec_idx = DECISION_VARS.index(fname)
                full_X[:, i] = decisions[:, dec_idx]
            else:
                # Use context value
                full_X[:, i] = context_row[i]
        return self.predict(full_X)

    def get_feature_importances(self, top_n: int = 15) -> Dict[str, float]:
        """Extract feature importances from the XGBoost sub-models.

        Averages the per-target feature importances across all outputs
        to produce a single unified ranking.

        Parameters
        ----------
        top_n : int
            Number of top features to return.

        Returns
        -------
        Dict[str, float]
            Feature name → importance score (normalized to sum to 1.0).
        """
        if not self._is_fitted:
            return {}

        # Each estimator in MultiOutputRegressor has its own feature_importances_
        all_importances = np.zeros(len(self.feature_names))
        n_estimators = len(self.model.estimators_)

        for estimator in self.model.estimators_:
            imp = estimator.feature_importances_
            all_importances += imp

        # Average across targets
        all_importances /= max(n_estimators, 1)

        # Normalize
        total = all_importances.sum()
        if total > 0:
            all_importances /= total

        # Build sorted dict
        feat_imp = {
            self.feature_names[i]: float(all_importances[i])
            for i in range(len(self.feature_names))
        }
        sorted_imp = dict(
            sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        )
        return sorted_imp


# ─────────────────────────────────────────────────────────────────────────────
# 2. NSGA-II MULTI-OBJECTIVE OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
class NSGAII:
    """Custom NSGA-II implementation for offline Pareto optimization.

    Objectives:
      - Maximize Tablet_Weight  (converted to minimization: -weight)
      - Minimize Power_Consumption_kW

    Constraints:
      - Friability ∈ [0.1, 1.0]
      - Hardness   ∈ [4.0, 10.0]

    Uses the XGBoost surrogate as the fast fitness evaluator.
    """

    def __init__(
        self,
        surrogate: SurrogateModel,
        context_row: np.ndarray,
        pop_size: int = NSGA_POP_SIZE,
        n_gen: int = NSGA_GENERATIONS,
        crossover_rate: float = NSGA_CROSSOVER_RATE,
        mutation_rate: float = NSGA_MUTATION_RATE,
        seed: int = 42,
    ) -> None:
        self.surrogate = surrogate
        self.context_row = context_row
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.rng = np.random.default_rng(seed)

        # Decision variable bounds as arrays
        self.lb = np.array([DECISION_BOUNDS[v][0] for v in DECISION_VARS])
        self.ub = np.array([DECISION_BOUNDS[v][1] for v in DECISION_VARS])
        self.n_vars = len(DECISION_VARS)

    def _init_population(self) -> np.ndarray:
        """Initialize population with Latin Hypercube-like sampling."""
        pop = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.n_vars))
        return pop

    def _evaluate(self, pop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate objectives and constraint violations.

        Returns
        -------
        objectives : np.ndarray, shape (pop_size, 2)
            Obj1 = -Tablet_Weight (minimization), Obj2 = Power_Consumption_kW
        violations : np.ndarray, shape (pop_size,)
            Sum of constraint violations (0 = feasible).
        """
        predictions = self.surrogate.predict_from_decisions(pop, self.context_row)
        # predictions columns match surrogate.target_names
        target_map = {t: i for i, t in enumerate(self.surrogate.target_names)}

        # Objectives
        tw_idx = target_map.get("Tablet_Weight", None)
        pw_idx = target_map.get("Power_Consumption_kW", None)

        obj1 = -predictions[:, tw_idx] if tw_idx is not None else np.zeros(len(pop))
        obj2 = predictions[:, pw_idx] if pw_idx is not None else np.zeros(len(pop))
        objectives = np.column_stack([obj1, obj2])

        # Constraints
        violations = np.zeros(len(pop))
        for constraint_name, (lo, hi) in QUALITY_CONSTRAINTS.items():
            if constraint_name in target_map:
                idx = target_map[constraint_name]
                vals = predictions[:, idx]
                violations += np.maximum(0, lo - vals)  # penalty for below lower
                violations += np.maximum(0, vals - hi)  # penalty for above upper

        return objectives, violations

    def _fast_non_dominated_sort(
        self, objectives: np.ndarray, violations: np.ndarray
    ) -> List[List[int]]:
        """Non-dominated sorting with constraint handling.

        Feasible solutions always dominate infeasible ones.
        Among infeasible, lower violation is preferred.
        """
        n = len(objectives)
        domination_count = np.zeros(n, dtype=int)
        dominated_set: List[List[int]] = [[] for _ in range(n)]
        fronts: List[List[int]] = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                dom = self._dominates(i, j, objectives, violations)
                if dom == 1:  # i dominates j
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif dom == -1:  # j dominates i
                    dominated_set[j].append(i)
                    domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            next_front: List[int] = []
            for i in fronts[k]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            k += 1
            fronts.append(next_front)

        return [f for f in fronts if f]  # remove empty fronts

    @staticmethod
    def _dominates(
        i: int, j: int, objectives: np.ndarray, violations: np.ndarray
    ) -> int:
        """Check if solution i dominates j (constraint-aware).

        Returns: 1 if i dominates j, -1 if j dominates i, 0 otherwise.
        """
        vi, vj = violations[i], violations[j]

        # Feasibility-first: feasible always dominates infeasible
        if vi == 0 and vj > 0:
            return 1
        if vj == 0 and vi > 0:
            return -1
        # Both infeasible: prefer lower violation
        if vi > 0 and vj > 0:
            return 1 if vi < vj else (-1 if vj < vi else 0)

        # Both feasible: standard Pareto dominance (minimize both objectives)
        oi, oj = objectives[i], objectives[j]
        if np.all(oi <= oj) and np.any(oi < oj):
            return 1
        if np.all(oj <= oi) and np.any(oj < oi):
            return -1
        return 0

    def _crowding_distance(self, front: List[int], objectives: np.ndarray) -> np.ndarray:
        """Calculate crowding distance for solutions in a front."""
        n = len(front)
        distances = np.zeros(n)
        if n <= 2:
            distances[:] = np.inf
            return distances

        for m in range(objectives.shape[1]):
            sorted_idx = np.argsort(objectives[front, m])
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            obj_range = (objectives[front[sorted_idx[-1]], m]
                         - objectives[front[sorted_idx[0]], m])
            if obj_range < 1e-12:
                continue
            for k in range(1, n - 1):
                distances[sorted_idx[k]] += (
                    objectives[front[sorted_idx[k + 1]], m]
                    - objectives[front[sorted_idx[k - 1]], m]
                ) / obj_range

        return distances

    def _tournament_select(
        self, pop: np.ndarray, ranks: np.ndarray, crowding: np.ndarray
    ) -> np.ndarray:
        """Binary tournament selection based on rank then crowding distance."""
        n = len(pop)
        selected = np.empty_like(pop)
        for i in range(n):
            a, b = self.rng.integers(0, n, size=2)
            if ranks[a] < ranks[b]:
                winner = a
            elif ranks[b] < ranks[a]:
                winner = b
            else:
                winner = a if crowding[a] > crowding[b] else b
            selected[i] = pop[winner]
        return selected

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX) with η=20."""
        eta = 20.0
        child1, child2 = parent1.copy(), parent2.copy()

        if self.rng.random() > self.crossover_rate:
            return child1, child2

        for j in range(self.n_vars):
            if self.rng.random() < 0.5:
                if abs(parent1[j] - parent2[j]) > 1e-14:
                    if parent1[j] < parent2[j]:
                        y1, y2 = parent1[j], parent2[j]
                    else:
                        y1, y2 = parent2[j], parent1[j]

                    beta = 1.0 + 2.0 * (y1 - self.lb[j]) / (y2 - y1 + 1e-14)
                    alpha = 2.0 - beta ** (-(eta + 1.0))
                    rand = self.rng.random()
                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))

                    child1[j] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                    child2[j] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                    child1[j] = np.clip(child1[j], self.lb[j], self.ub[j])
                    child2[j] = np.clip(child2[j], self.lb[j], self.ub[j])

        return child1, child2

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Polynomial mutation with η=20."""
        eta = 20.0
        child = individual.copy()
        for j in range(self.n_vars):
            if self.rng.random() < self.mutation_rate:
                delta_max = self.ub[j] - self.lb[j]
                delta = (child[j] - self.lb[j]) / (delta_max + 1e-14)
                rand = self.rng.random()
                if rand < 0.5:
                    xy = 1.0 - delta
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                    deltaq = val ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    xy = 1.0 - (1.0 - delta)
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                    deltaq = 1.0 - val ** (1.0 / (eta + 1.0))
                child[j] += deltaq * delta_max
                child[j] = np.clip(child[j], self.lb[j], self.ub[j])
        return child

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute the NSGA-II optimization loop.

        Returns
        -------
        pareto_decisions : np.ndarray
            Decision variables of Pareto-front solutions.
        pareto_objectives : np.ndarray
            Objective values (−Tablet_Weight, Power_Consumption_kW).
        pareto_predictions : np.ndarray
            Full target predictions for Pareto solutions.
        """
        print(f"\n[NSGA-II] Starting optimization:")
        print(f"  Population : {self.pop_size}")
        print(f"  Generations: {self.n_gen}")
        print(f"  Variables  : {self.n_vars}")

        pop = self._init_population()

        for gen in range(self.n_gen):
            # Evaluate
            objectives, violations = self._evaluate(pop)

            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(objectives, violations)

            # Assign ranks and crowding distances
            ranks = np.zeros(len(pop), dtype=int)
            crowding = np.zeros(len(pop))
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
                cd = self._crowding_distance(front, objectives)
                for k, idx in enumerate(front):
                    crowding[idx] = cd[k]

            # Progress logging
            n_feasible = (violations == 0).sum()
            if gen % 30 == 0 or gen == self.n_gen - 1:
                print(f"  Gen {gen:4d}/{self.n_gen}: "
                      f"Pareto front size = {len(fronts[0])}, "
                      f"Feasible = {n_feasible}/{self.pop_size}")

            # Selection + offspring
            selected = self._tournament_select(pop, ranks, crowding)
            offspring = np.empty_like(pop)
            for i in range(0, self.pop_size, 2):
                p1 = selected[i]
                p2 = selected[min(i + 1, self.pop_size - 1)]
                c1, c2 = self._crossover(p1, p2)
                offspring[i] = self._mutate(c1)
                if i + 1 < self.pop_size:
                    offspring[i + 1] = self._mutate(c2)

            # Combine parent + offspring (μ+λ strategy)
            combined = np.vstack([pop, offspring])
            combined_obj, combined_vio = self._evaluate(combined)
            combined_fronts = self._fast_non_dominated_sort(
                combined_obj, combined_vio
            )

            # Select next generation
            new_pop = []
            for front in combined_fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend(front)
                else:
                    remaining = self.pop_size - len(new_pop)
                    cd = self._crowding_distance(front, combined_obj)
                    sorted_by_cd = np.argsort(-cd)
                    for idx in sorted_by_cd[:remaining]:
                        new_pop.append(front[idx])
                    break
            pop = combined[new_pop]

        # ── Extract final Pareto front ──────────────────────────────────
        final_obj, final_vio = self._evaluate(pop)
        final_fronts = self._fast_non_dominated_sort(final_obj, final_vio)
        pareto_indices = final_fronts[0]

        pareto_decisions = pop[pareto_indices]
        pareto_objectives = final_obj[pareto_indices]

        # Get full predictions for the Pareto solutions
        pareto_predictions = self.surrogate.predict_from_decisions(
            pareto_decisions, self.context_row
        )

        # Filter to only feasible solutions
        feasible_mask = final_vio[pareto_indices] == 0
        if feasible_mask.sum() > 0:
            pareto_decisions = pareto_decisions[feasible_mask]
            pareto_objectives = pareto_objectives[feasible_mask]
            pareto_predictions = pareto_predictions[feasible_mask]
            print(f"\n[NSGA-II] Final Pareto front: {len(pareto_decisions)} "
                  f"feasible solutions")
        else:
            print(f"\n[NSGA-II] ⚠ No fully feasible solutions found, "
                  f"returning {len(pareto_decisions)} best solutions")

        return pareto_decisions, pareto_objectives, pareto_predictions


# ─────────────────────────────────────────────────────────────────────────────
# 3. GOLDEN SIGNATURES GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_golden_signatures(
    training_df: Optional[pd.DataFrame] = None,
    n_contexts: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a massive dataset of Pareto-optimal 'Golden Signatures'.

    Runs the NSGA-II optimizer across multiple sampled historical contexts
    to produce a rich set of optimal machine settings for proxy training.

    Parameters
    ----------
    training_df : pd.DataFrame, optional
        Merged training dataset. If None, builds from data_layer.
    n_contexts : int
        Number of historical contexts to optimize over.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Golden Signatures with columns:
        [context_features..., decision_variables..., predicted_targets...]
    """
    if training_df is None:
        training_df = build_training_dataset()

    # ── Train surrogate ─────────────────────────────────────────────────
    surrogate = SurrogateModel(seed=seed)
    surrogate.fit(training_df)

    # ── Sample contexts from training data ──────────────────────────────
    rng = np.random.default_rng(seed)
    n_contexts = min(n_contexts, len(training_df))
    context_indices = rng.choice(len(training_df), size=n_contexts, replace=False)

    all_signatures: List[Dict[str, object]] = []

    for i, ctx_idx in enumerate(context_indices):
        context_full = training_df.iloc[ctx_idx]
        context_row = training_df[surrogate.feature_names].iloc[ctx_idx].values

        print(f"\n{'─' * 60}")
        print(f"Context {i + 1}/{n_contexts}: Batch {context_full['Batch_ID']}")

        # Run NSGA-II for this context
        optimizer = NSGAII(surrogate, context_row, seed=seed + i)
        decisions, objectives, predictions = optimizer.run()

        # Build signature records
        for j in range(len(decisions)):
            record: Dict[str, object] = {}

            # Context features (non-decision, non-target)
            for feat_name in surrogate.feature_names:
                if feat_name not in DECISION_VARS:
                    record[f"ctx_{feat_name}"] = context_row[
                        surrogate.feature_names.index(feat_name)
                    ]

            # Optimal decision variables
            for k, var_name in enumerate(DECISION_VARS):
                record[var_name] = decisions[j, k]

            # Predicted targets
            for k, target_name in enumerate(surrogate.target_names):
                record[f"pred_{target_name}"] = predictions[j, k]

            # Pareto objective values
            record["obj_neg_Tablet_Weight"] = objectives[j, 0]
            record["obj_Power_Consumption"] = objectives[j, 1]

            all_signatures.append(record)

    golden_df = pd.DataFrame(all_signatures)
    print(f"\n{'=' * 60}")
    print(f"[GOLDEN SIGNATURES] Generated {len(golden_df)} Pareto-optimal solutions "
          f"across {n_contexts} contexts")
    print(f"{'=' * 60}")

    return golden_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     OFFLINE OPTIMIZER — Phase 1: Core Engine            ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # Build training data
    training_data = build_training_dataset()

    # Generate Golden Signatures
    golden_sigs = generate_golden_signatures(
        training_df=training_data,
        n_contexts=20,   # Run across 20 historical scenarios
        seed=42,
    )

    # ── Save Golden Signatures ──────────────────────────────────────────
    output_path = os.path.join(DATA_DIR, "golden_signatures.csv")
    golden_sigs.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(golden_sigs)} Golden Signatures to: {output_path}")

    # ── Summary statistics ──────────────────────────────────────────────
    print(f"\nGolden Signatures Summary:")
    print(f"  Shape: {golden_sigs.shape}")
    for var in DECISION_VARS:
        if var in golden_sigs.columns:
            print(f"  {var:25s}: [{golden_sigs[var].min():.2f}, "
                  f"{golden_sigs[var].max():.2f}]")

    for target in TARGET_COLS:
        pred_col = f"pred_{target}"
        if pred_col in golden_sigs.columns:
            print(f"  pred_{target:20s}: [{golden_sigs[pred_col].min():.2f}, "
                  f"{golden_sigs[pred_col].max():.2f}]")
