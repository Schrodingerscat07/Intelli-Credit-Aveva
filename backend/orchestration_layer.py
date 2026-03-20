"""
==============================================================================
ORCHESTRATION LAYER  --  Phase 2: Agentic Orchestration
==============================================================================
LangGraph state-machine controller that routes factory data, retrieves vector
memory from Qdrant, executes the Phase-1 PyTorch Optimization Proxy, manages
a Human-in-the-Loop (HITL) approval gate, and feeds outcomes back into the
vector memory for continuous learning.

Author : Core Engine Team
Version: 1.0.0

Architecture:
  +-----------+     +--------------+     +------+     +-----------+
  | data_     |---->| proxy_caller |---->| HITL |---->| execution |
  | router    |     |   (Brain)    |     | Gate |     |   node    |
  +-----------+     +--------------+     +------+     +-----------+
       |                  |                                 |
       v                  v                                 v
    [Qdrant]        [LangSmith +              [MCP Tool + Qdrant
     Vector          Openlayer                 Continuous
     Memory          Tracing]                  Learning]
==============================================================================
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import logging
import warnings
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# ── LangGraph ───────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# ── LangSmith Observability ────────────────────────────────────────────
from langsmith import traceable

# ── Qdrant Vector Database ─────────────────────────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

# NOTE: SentenceTransformer REMOVED -- text embeddings destroy
# mathematical relationships in numerical telemetry data.
# We use raw L2-normalized numerical vectors directly in Qdrant.

# ── Phase 1 Imports ────────────────────────────────────────────────────
from data_layer import build_training_dataset, DATA_DIR, SENSOR_COLS
from offline_optimizer import DECISION_VARS, DECISION_BOUNDS, TARGET_COLS, SurrogateModel
from model_layer import (
    OptimizationProxy,
    RepairLayer,
    InferenceEngine,
    DEVICE,
)

# ── Phase 2+ Imports: New Feature Modules ──────────────────────────────
from carbon_tracker import CarbonTracker, calculate_carbon
from batch_history import BatchHistoryStore
from energy_analyzer import EnergyPatternAnalyzer
from decision_memory import DecisionMemory

# ── PyTorch / sklearn ──────────────────────────────────────────────────
import torch
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("orchestration")

# =====================================================================
# CONSTANTS
# =====================================================================
QDRANT_COLLECTION = "golden_signatures"
# EMBEDDING_DIM is set dynamically from the number of context columns
# at initialization time, not hardcoded for a text model.

# Simulated physical constraints for the MCP tool
MCP_TOOL_NAME = "execute_machine_parameters"


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for
    JSON/msgpack serialization compatibility with LangGraph checkpoints.
    Also sanitizes inf/nan which are not JSON-compliant."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# =====================================================================
# 1. PYDANTIC STATE  --  Strictly typed graph state
# =====================================================================
class ManufacturingState(BaseModel):
    """Strictly typed LangGraph state for the manufacturing optimization
    workflow.

    Every field tracks a dimension of the orchestration pipeline, from
    incoming telemetry through to execution outcome.
    """
    # ── Identifiers ─────────────────────────────────────────────────
    batch_id: str = Field(
        default="", description="Current batch being optimized"
    )
    run_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique run identifier for tracing",
    )

    # ── Node 1: Data Router outputs ─────────────────────────────────
    current_telemetry: Dict[str, Any] = Field(
        default_factory=dict,
        description="Live / simulated telemetry readings",
    )
    historical_baseline: Dict[str, Any] = Field(
        default_factory=dict,
        description="Closest Golden Signature from Qdrant vector memory",
    )
    baseline_score: float = Field(
        default=0.0,
        description="Cosine similarity of the retrieved baseline",
    )

    # ── Node 2: Proxy Caller outputs ───────────────────────────────
    proposed_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optimal machine settings from PyTorch proxy + Repair",
    )
    raw_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pre-repair raw neural network outputs (for audit)",
    )

    # ── HITL Gate ───────────────────────────────────────────────────
    human_approved: bool = Field(
        default=False,
        description="Whether a human operator approved the proposed settings",
    )
    human_feedback: str = Field(
        default="",
        description="Optional textual feedback from human reviewer",
    )

    # ── Node 3: Execution outputs ──────────────────────────────────
    execution_status: str = Field(
        default="pending",
        description="One of: pending | approved | rejected | executed | failed",
    )
    simulated_outcome: Dict[str, Any] = Field(
        default_factory=dict,
        description="Simulated production outcome after execution",
    )
    quality_delta: float = Field(
        default=0.0,
        description="Improvement in yield-to-energy ratio vs baseline",
    )
    qdrant_updated: bool = Field(
        default=False,
        description="Whether Qdrant was updated with a new golden signature",
    )

    # ── NEW: Carbon Emissions ──────────────────────────────────────
    carbon_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per-batch carbon emission calculations",
    )

    # ── NEW: Energy Pattern Analysis ───────────────────────────────
    energy_anomalies: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detected energy consumption anomalies vs baseline",
    )
    asset_health_score: float = Field(
        default=100.0,
        description="Overall asset health score (0-100)",
    )
    energy_recommendations: List[str] = Field(
        default_factory=list,
        description="Maintenance recommendations from energy analysis",
    )

    # ── NEW: Decision Memory ───────────────────────────────────────
    past_decision_warnings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Warnings about similar past HITL decisions",
    )

    # ── NEW: Optimization Priorities ───────────────────────────────
    optimization_priorities: Dict[str, Any] = Field(
        default_factory=lambda: {
            "objective_primary": "Tablet_Weight",
            "objective_secondary": "Power_Consumption_kW",
            "priority_value": 50,
            "mode": "yield_vs_energy",
        },
        description="User-configurable optimization objective priorities",
    )

    # ── Phase 2: Qdrant Fallback ───────────────────────────────────
    no_confident_match: bool = Field(
        default=False,
        description="True if Qdrant returned no confident match (score < 0.85)",
    )

    # ── Phase 2: Novelty Detection ─────────────────────────────────
    novelty_warning: Dict[str, Any] = Field(
        default_factory=dict,
        description="Novelty detection result from Mahalanobis distance check",
    )

    # ── Phase 2: Prediction Intervals ──────────────────────────────
    prediction_intervals: Dict[str, Any] = Field(
        default_factory=dict,
        description="Uncertainty quantification: lower/upper bounds per target",
    )

    # ── Phase 2: Drift Detection ───────────────────────────────────
    retraining_alert: bool = Field(
        default=False,
        description="True if model drift detected (3+ consecutive OOB batches)",
    )
# =====================================================================
# 2. VECTOR MEMORY  --  Qdrant Manager (Raw Numerical Vectors)
# =====================================================================
class VectorMemory:
    """Manages the Qdrant in-memory vector database for Golden Signature
    retrieval and continuous learning updates.

    CRITICAL DESIGN DECISION:
      We use raw L2-normalized numerical feature vectors directly as
      dense vector embeddings in Qdrant, NOT text-based transformers.
      This preserves the mathematical distance relationships in our
      factory telemetry (e.g., 45 deg C is close to 46 deg C), which
      would be destroyed by stringifying numbers and passing them
      through a language model.
    """

    def __init__(self, context_cols: List[str]) -> None:
        log.info("Initializing Qdrant in-memory vector database...")
        self.client = QdrantClient(":memory:")  # no external server needed
        self.context_cols = context_cols
        self.vector_dim = len(context_cols)
        self.scaler = StandardScaler()  # fitted during ingestion
        self._scaler_fitted = False

        # Create collection with dimension = number of context features
        self.client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=self.vector_dim,
                distance=Distance.COSINE,
            ),
        )
        log.info(
            "Qdrant collection '%s' created (dim=%d, cosine, raw numerical)",
            QDRANT_COLLECTION, self.vector_dim,
        )

    def _to_vector(self, features: Dict[str, Any]) -> List[float]:
        """Convert a feature dict to an L2-normalized numerical vector.

        Uses the same column ordering and StandardScaler as the ingested
        Golden Signatures so that cosine similarity is meaningful.
        """
        raw = np.array(
            [float(features.get(c, 0.0)) for c in self.context_cols],
            dtype=np.float64,
        ).reshape(1, -1)

        # Scale using the fitted scaler (REMOVED: scaling centers around 0, 
        # which breaks cosine similarity directionality for perturbed vectors)
        scaled = raw.flatten()

        # L2 normalize for cosine similarity
        norm = np.linalg.norm(scaled)
        if norm > 1e-12:
            scaled = scaled / norm

        return scaled.tolist()

    def ingest_golden_signatures(self, golden_df: pd.DataFrame) -> int:
        """Bulk-load Golden Signatures into Qdrant using raw numerical vectors.

        Parameters
        ----------
        golden_df : pd.DataFrame
            Golden Signatures CSV from offline_optimizer.

        Returns
        -------
        int
            Number of points ingested.
        """
        # ── Fit the StandardScaler on ALL context features ──────────
        available_ctx = [c for c in self.context_cols if c in golden_df.columns]
        X_ctx = golden_df[available_ctx].fillna(0.0).values.astype(np.float64)
        self.scaler.fit(X_ctx)
        self._scaler_fitted = True
        log.info("  Qdrant scaler fitted on %d context features", len(available_ctx))

        # ── Build points with raw numerical vectors ─────────────────
        points: List[PointStruct] = []
        decision_cols = [c for c in DECISION_VARS if c in golden_df.columns]
        pred_cols = [c for c in golden_df.columns if c.startswith("pred_")]

        for idx, row in golden_df.iterrows():
            # Build raw numerical vector from context columns
            feature_dict = {c: float(row.get(c, 0.0)) for c in self.context_cols}
            vector = self._to_vector(feature_dict)

            # Payload = decision vars + predictions (for retrieval)
            payload: Dict[str, Any] = {}
            for c in decision_cols:
                payload[c] = float(row[c])
            for c in pred_cols:
                payload[c] = float(row[c])
            # Also store context features in payload for proxy caller
            for c in available_ctx:
                payload[c] = float(row[c])
            payload["source"] = "nsga2_offline"

            points.append(PointStruct(
                id=int(idx) if isinstance(idx, (int, np.integer)) else hash(str(idx)) % (2**63),
                vector=vector,
                payload=payload,
            ))

        # Batch upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points[i:i + batch_size],
            )

        log.info(
            "Ingested %d Golden Signatures into Qdrant '%s' (numerical vectors)",
            len(points), QDRANT_COLLECTION,
        )
        return len(points)

    def query_nearest(
        self, telemetry: Dict[str, Any], top_k: int = 1
    ) -> Dict[str, Any]:
        """Find the closest historical Golden Signature using numerical
        cosine similarity on the raw feature vectors.

        Parameters
        ----------
        telemetry : Dict[str, Any]
            Current telemetry feature dict (same keys as context_cols).
        top_k : int
            Number of nearest neighbors to retrieve.

        Returns
        -------
        Dict[str, Any]
            Best matching Golden Signature payload + similarity score.
        """
        vector = self._to_vector(telemetry)

        results = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            limit=top_k,
        )

        if results.points:
            best = results.points[0]
            return {
                "payload": best.payload,
                "score": best.score,
                "id": best.id,
            }
        return {"payload": {}, "score": 0.0, "id": None}

    def upsert_new_signature(
        self, features: Dict[str, Any], settings: Dict[str, Any],
        outcome: Dict[str, Any],
    ) -> int:
        """Add a new learned Golden Signature to Qdrant (continuous learning).

        Parameters
        ----------
        features : Dict[str, Any]
            Context features that led to the good outcome.
        settings : Dict[str, Any]
            The machine settings that were applied.
        outcome : Dict[str, Any]
            The simulated/actual production outcome.

        Returns
        -------
        int
            The point ID of the new entry.
        """
        vector = self._to_vector(features)

        payload = {**settings, **{f"pred_{k}": v for k, v in outcome.items()}}
        payload["source"] = "continuous_learning"

        point_id = abs(hash(str(uuid.uuid4()))) % (2**63)
        self.client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        log.info(
            "Continuous Learning: upserted new signature (id=%d) into Qdrant",
            point_id,
        )
        return point_id


# =====================================================================
# 3. MCP TOOL DEFINITION  --  Secure Execution via MCP Protocol
# =====================================================================
class MCPToolExecutor:
    """Mock MCP (Model Context Protocol) client tool for simulating
    secure machine parameter execution.

    In production, this would connect to a real MCP server managing
    factory equipment via langchain-mcp-adapters. For the hackathon
    prototype, it simulates the execution and returns a synthetic
    production outcome.

    Integration point:
      from langchain_mcp_adapters.client import MultiServerMCPClient
      async with MultiServerMCPClient(server_configs) as client:
          tools = client.get_tools()
          # bind tools to agent
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self._execution_log: List[Dict[str, Any]] = []

    def execute_machine_parameters(
        self, settings: Dict[str, float], batch_id: str
    ) -> Dict[str, Any]:
        """Simulate pushing optimized settings to the factory floor.

        This is the MCP tool that would be exposed via:
          langchain_mcp_adapters for secure, protocol-compliant execution.

        Parameters
        ----------
        settings : Dict[str, float]
            Validated machine settings from the Optimization Proxy + Repair.
        batch_id : str
            The batch being produced.

        Returns
        -------
        Dict[str, Any]
            Simulated production outcome and execution metadata.
        """
        log.info(
            "[MCP TOOL] execute_machine_parameters called for batch %s",
            batch_id,
        )

        # ── Validate all settings are within physical bounds ────────
        for var, val in settings.items():
            if var in DECISION_BOUNDS:
                lo, hi = DECISION_BOUNDS[var]
                if val < lo - 0.01 or val > hi + 0.01:
                    raise ValueError(
                        f"MCP SAFETY VIOLATION: {var}={val:.2f} "
                        f"outside [{lo}, {hi}]"
                    )

        # ── Simulate production outcome ─────────────────────────────
        # Realistic simulation: higher compression + speed -> heavier tablets
        # Higher drying temp -> lower moisture -> higher friability
        base_weight = 200.0
        weight_factor = (
            settings.get("Compression_Force", 12.0) / 12.0
            * settings.get("Machine_Speed", 50.0) / 50.0
        )
        tablet_weight = base_weight * weight_factor * (1 + self.rng.normal(0, 0.02))

        hardness = 6.0 + self.rng.normal(0, 1.0)
        friability = 0.5 + settings.get("Drying_Temp", 60.0) / 200.0 + self.rng.normal(0, 0.1)
        power = (
            settings.get("Machine_Speed", 50.0) * 0.3
            + settings.get("Compression_Force", 12.0) * 0.5
            + self.rng.normal(0, 1.0)
        )

        outcome = _to_native({
            "Tablet_Weight": round(float(tablet_weight), 2),
            "Hardness": round(float(np.clip(hardness, 1, 12)), 2),
            "Friability": round(float(np.clip(friability, 0.05, 2.0)), 4),
            "Power_Consumption_kW": round(float(max(power, 5.0)), 2),
        })

        record = {
            "batch_id": batch_id,
            "settings": settings,
            "outcome": outcome,
            "timestamp": time.time(),
            "status": "executed",
        }
        self._execution_log.append(record)

        log.info(
            "[MCP TOOL] Execution complete: Tablet_Weight=%.1f, Power=%.1f kW",
            outcome["Tablet_Weight"], outcome["Power_Consumption_kW"],
        )
        return record


# =====================================================================
# 4. OPENLAYER CALLBACK (Observability)
# =====================================================================
class OpenlayerMonitor:
    """Lightweight Openlayer-compatible callback handler for monitoring
    proxy outputs and flagging hallucinated parameters.

    In production, connect via:
      from openlayer.lib import trace as openlayer_trace
      @openlayer_trace(...)
      def monitored_function(...): ...

    For the prototype, we implement local anomaly detection that mirrors
    what Openlayer would flag.
    """

    def __init__(self) -> None:
        self._traces: List[Dict[str, Any]] = []

    def log_proxy_output(
        self, raw: Dict[str, float], repaired: Dict[str, float],
        batch_id: str,
    ) -> Dict[str, Any]:
        """Log and validate proxy outputs for hallucination detection.

        Parameters
        ----------
        raw : Dict[str, float]
            Pre-repair neural network outputs.
        repaired : Dict[str, float]
            Post-repair physically-feasible outputs.
        batch_id : str
            Batch identifier.

        Returns
        -------
        Dict[str, Any]
            Trace record with anomaly flags.
        """
        anomalies: List[str] = []
        for var in DECISION_VARS:
            if var in raw and var in repaired:
                if abs(raw[var] - repaired[var]) > 0.01:
                    anomalies.append(
                        f"{var}: raw={raw[var]:.2f} -> repaired={repaired[var]:.2f}"
                    )

        trace = {
            "batch_id": batch_id,
            "raw_output": raw,
            "repaired_output": repaired,
            "n_clamped": len(anomalies),
            "anomalies": anomalies,
            "timestamp": time.time(),
            "hallucination_detected": len(anomalies) > 0,
        }
        self._traces.append(trace)

        if anomalies:
            log.warning(
                "[OPENLAYER] %d parameters clamped by Repair Layer for batch %s",
                len(anomalies), batch_id,
            )
            for a in anomalies:
                log.warning("  -> %s", a)
        else:
            log.info("[OPENLAYER] All proxy outputs within bounds for batch %s", batch_id)

        return trace


# =====================================================================
# 5. GRAPH NODES
# =====================================================================

# ── Shared resources (initialized at graph build time) ──────────────
_vector_memory: Optional[VectorMemory] = None
_mcp_executor: Optional[MCPToolExecutor] = None
_openlayer: Optional[OpenlayerMonitor] = None
_proxy_model: Optional[OptimizationProxy] = None
_input_scaler: Optional[StandardScaler] = None
_output_scaler: Optional[StandardScaler] = None

# ── NEW: Feature modules ────────────────────────────────────────────
_carbon_tracker: Optional[CarbonTracker] = None
_batch_history: Optional[BatchHistoryStore] = None
_energy_analyzer: Optional[EnergyPatternAnalyzer] = None
_decision_memory: Optional[DecisionMemory] = None
_surrogate_model: Optional[SurrogateModel] = None

# ── NEW: Global priorities (updated by frontend) ────────────────────
_current_priorities: Dict[str, Any] = {
    "objective_primary": "Tablet_Weight",
    "objective_secondary": "Power_Consumption_kW",
    "priority_value": 50,
    "mode": "yield_vs_energy",
}

# ── Phase 2: Drift Detection counter ────────────────────────────────
_drift_counter: int = 0  # consecutive batches outside prediction interval

# ── NEW: Regulatory targets ─────────────────────────────────────────
_regulatory_targets: Dict[str, Any] = {
    "max_carbon_per_batch_kg": 25.0,
    "max_power_per_batch_kwh": 50.0,
    "min_yield_pct": 90.0,
    "min_hardness": 4.0,
    "max_friability": 1.0,
    "emission_factor_name": "india_grid",
}


def _get_context_cols(golden_df: pd.DataFrame) -> List[str]:
    """Extract context column names from Golden Signatures."""
    return [c for c in golden_df.columns if c.startswith("ctx_")]


# ─────────────────────────────────────────────────────────────────────
# NODE 1: Data Router & Memory Retrieval
# ─────────────────────────────────────────────────────────────────────
@traceable(name="data_router_node", run_type="chain")
def data_router_node(state: ManufacturingState) -> dict:
    """Receive incoming telemetry and query Qdrant for the closest
    matching historical Golden Signature baseline.

    This node:
      1. Accepts the current telemetry from the state
      2. Embeds it and queries the Qdrant vector DB
      3. Retrieves the best historical baseline
      4. Updates state with baseline context

    API Hook (Phase 3 React Dashboard):
      POST /api/telemetry  ->  triggers this node with live sensor data
    """
    log.info("=" * 60)
    log.info("NODE 1: Data Router & Memory Retrieval")
    log.info("  Batch: %s", state.batch_id)
    log.info("  Telemetry keys: %d", len(state.current_telemetry))

    # Query Qdrant for nearest Golden Signature
    result = _vector_memory.query_nearest(state.current_telemetry, top_k=1)

    baseline = result["payload"]
    score = result["score"]

    log.info("  Qdrant match score: %.4f", score)
    log.info(
        "  Baseline source: %s",
        baseline.get("source", "unknown"),
    )

    # ── Phase 2: Qdrant Fallback (Feature 12) ─────────────────────
    QDRANT_CONFIDENCE_THRESHOLD = 0.85
    no_confident_match = score < QDRANT_CONFIDENCE_THRESHOLD
    if no_confident_match:
        log.warning("  ⚠ LOW QDRANT CONFIDENCE: score %.4f < %.2f threshold",
                    score, QDRANT_CONFIDENCE_THRESHOLD)
        log.warning("  → No close historical scenario found — recommend manual review")
    else:
        log.info("  ✓ Qdrant match above confidence threshold")

    # ── NEW: Energy Pattern Analysis ──────────────────────────────
    energy_result = _energy_analyzer.analyze_patterns(
        state.current_telemetry, baseline
    )
    log.info("  Asset Health Score: %.1f%%", energy_result["asset_health_score"])
    log.info("  Energy Anomalies: %d detected", energy_result["anomaly_count"])
    for rec in energy_result["recommendations"]:
        log.info("    → %s", rec)

    return _to_native({
        "historical_baseline": baseline,
        "baseline_score": score,
        "no_confident_match": no_confident_match,
        "energy_anomalies": energy_result["anomalies"],
        "asset_health_score": energy_result["asset_health_score"],
        "energy_recommendations": energy_result["recommendations"],
    })


# ─────────────────────────────────────────────────────────────────────
# NODE 2: Proxy Caller (The Brain)
# ─────────────────────────────────────────────────────────────────────
@traceable(name="proxy_caller_node", run_type="chain")
def proxy_caller_node(state: ManufacturingState) -> dict:
    """Pass state context to the Phase 1 PyTorch Optimization Proxy
    and retrieve instantly generated, physically repaired settings.

    This node:
      1. Extracts context features from state
      2. Runs the neural proxy forward pass
      3. Applies the Repair Layer
      4. Logs to Openlayer for hallucination monitoring
      5. Returns proposed optimal settings

    Wrapped with @traceable for LangSmith observability.
    OpenlayerCallbackHandler monitors for hallucinated parameters.

    API Hook (Phase 3 React Dashboard):
      GET /api/proposed-settings  ->  returns this node's output
    """
    log.info("=" * 60)
    log.info("NODE 2: Proxy Caller (The Brain)")

    # ── Build context feature vector from baseline ──────────────────
    baseline = state.historical_baseline
    context_cols = [k for k in baseline.keys()
                    if k.startswith("ctx_") or k.startswith("pred_")]

    if not context_cols:
        # Fallback: use all numeric baseline keys as context
        context_cols = [k for k in baseline.keys()
                        if k not in ("source",) and isinstance(baseline.get(k), (int, float))]

    context_values = np.array(
        [float(baseline.get(c, 0.0)) for c in context_cols],
        dtype=np.float32,
    ).reshape(1, -1)

    # Ensure context matches proxy input dimension
    proxy_input_dim = _proxy_model.backbone[0].in_features
    if context_values.shape[1] < proxy_input_dim:
        padding = np.zeros((1, proxy_input_dim - context_values.shape[1]), dtype=np.float32)
        context_values = np.hstack([context_values, padding])
    elif context_values.shape[1] > proxy_input_dim:
        context_values = context_values[:, :proxy_input_dim]

    # ── Scale and run proxy ─────────────────────────────────────────
    start_ns = time.perf_counter_ns()

    X_scaled = _input_scaler.transform(context_values).astype(np.float32)
    X_tensor = torch.tensor(X_scaled).to(DEVICE)

    _proxy_model.eval()
    with torch.no_grad():
        raw_scaled = _proxy_model.backbone(X_tensor)
        raw_np = _output_scaler.inverse_transform(raw_scaled.cpu().numpy())
        raw_tensor = torch.tensor(raw_np.astype(np.float32)).to(DEVICE)
        repaired = _proxy_model.repair(raw_tensor).cpu().numpy()

    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

    # ── Build output dicts ──────────────────────────────────────────
    raw_settings = {var: float(raw_np[0, i]) for i, var in enumerate(DECISION_VARS)}
    proposed = {var: float(repaired[0, i]) for i, var in enumerate(DECISION_VARS)}

    log.info("  Proxy inference: %.2f ms", elapsed_ms)
    for var in DECISION_VARS:
        lo, hi = DECISION_BOUNDS[var]
        status = "OK" if lo <= proposed[var] <= hi else "CLAMPED"
        log.info(
            "    %s: %.2f [%.1f, %.1f] %s",
            var, proposed[var], lo, hi, status,
        )

    # ── Openlayer monitoring ────────────────────────────────────────
    _openlayer.log_proxy_output(raw_settings, proposed, state.batch_id)

    state_additions = {
        "raw_settings": raw_settings,
        "proposed_settings": proposed,
    }

    # ── Feature 12: Qdrant Fallback ──────────────────────────────────
    no_confident_match = state.baseline_score < 0.85
    if no_confident_match:
        log.warning("  ⚠️ Low Qdrant Match Score (%.4f) - Triggering Fallback Warning", state.baseline_score)
    state_additions["no_confident_match"] = no_confident_match

    # ── Feature 10: Novelty Detection from Surrogate ──────────────────
    if _surrogate_model and _surrogate_model._is_fitted:
        try:
            novelty_result = _surrogate_model.check_novelty(context_values)
            state_additions["novelty_warning"] = novelty_result
            if novelty_result.get("is_novel", False):
                log.warning("  ⚠️ Novelty detected! Mahalanobis distance %.2f > threshold %.2f",
                            novelty_result["distance"], novelty_result["threshold"])
            else:
                log.info("  ✓ Input within known training distribution (dist=%.2f)",
                         novelty_result.get("distance", 0))
        except Exception as e:
            log.error("  Novelty check failed: %s", e)

    return _to_native(state_additions)


# ─────────────────────────────────────────────────────────────────────
# HITL GATE: Human-in-the-Loop Interrupt
# ─────────────────────────────────────────────────────────────────────
@traceable(name="hitl_gate_node", run_type="chain")
def hitl_gate_node(state: ManufacturingState) -> dict:
    """Human-in-the-Loop approval gate.

    This node uses LangGraph's interrupt() to pause execution and
    surface the proposed settings for human review. The graph will
    NOT proceed to the execution node until a human approves or
    rejects the proposed settings.

    In production, the React dashboard calls:
      POST /api/approve  with body { "approved": true/false }
    which resumes the graph via Command(resume={"approved": True}).

    API Hooks (Phase 3 React Dashboard):
      GET  /api/pending-approval  ->  returns proposed_settings + context
      POST /api/approve           ->  resumes graph with approval decision
    """
    log.info("=" * 60)
    log.info("HITL GATE: Requesting human approval")
    log.info("  Proposed settings for batch %s:", state.batch_id)
    for var, val in state.proposed_settings.items():
        log.info("    %s = %.2f", var, val)

    # ── NEW: Check decision memory for similar past decisions ──────
    decision_warnings = _decision_memory.get_warnings(state.proposed_settings)
    if decision_warnings:
        log.info("  ⚠️ Decision Memory warnings:")
        for w in decision_warnings:
            log.info("    → %s (Feedback: %s)", w["message"], w["feedback"])

    # ── INTERRUPT: Pause graph execution here ───────────────────────
    human_decision = interrupt({
        "message": "Please review and approve the proposed machine settings.",
        "batch_id": state.batch_id,
        "proposed_settings": state.proposed_settings,
        "baseline_score": state.baseline_score,
        "past_decision_warnings": decision_warnings,
        "instructions": (
            "POST /api/approve with {'approved': true} to proceed, "
            "or {'approved': false, 'feedback': '...'} to reject."
        ),
    })

    # ── Process the human's response ────────────────────────────────
    approved = human_decision.get("approved", False)
    feedback = human_decision.get("feedback", "")

    if approved:
        log.info("  APPROVED by human operator")
        return {
            "human_approved": True,
            "human_feedback": feedback,
            "execution_status": "approved",
            "past_decision_warnings": decision_warnings,
        }
    else:
        log.info("  REJECTED by human operator. Feedback: %s", feedback)
        # Log rejected decision immediately
        _decision_memory.log_decision(
            batch_id=state.batch_id,
            proposed_settings=state.proposed_settings,
            approved=False,
            feedback=feedback,
        )
        return {
            "human_approved": False,
            "human_feedback": feedback,
            "execution_status": "rejected",
            "past_decision_warnings": decision_warnings,
        }


# ─────────────────────────────────────────────────────────────────────
# NODE 3: Execution & Continuous Learning
# ─────────────────────────────────────────────────────────────────────
@traceable(name="execution_node", run_type="chain")
def execution_node(state: ManufacturingState) -> dict:
    """Execute approved settings and feed outcomes back for continuous
    learning.

    This node:
      1. Checks human_approved flag
      2. Uses the MCP tool to simulate factory execution
      3. Compares outcome against historical baseline
      4. If yield-to-energy ratio improved, upserts to Qdrant

    Secure Execution via MCP:
      The execute_machine_parameters tool is defined following the
      Model Context Protocol specification. In production, this would
      be exposed via langchain-mcp-adapters' MultiServerMCPClient.

    API Hook (Phase 3 React Dashboard):
      GET /api/execution-result  ->  returns outcome + quality delta
    """
    log.info("=" * 60)
    log.info("NODE 3: Execution & Continuous Learning")

    if not state.human_approved:
        log.info("  Execution SKIPPED: human did not approve")
        return {
            "execution_status": "rejected",
            "simulated_outcome": {},
            "quality_delta": 0.0,
        }

    # ── Execute via MCP Tool ────────────────────────────────────────
    try:
        result = _mcp_executor.execute_machine_parameters(
            settings=state.proposed_settings,
            batch_id=state.batch_id,
        )
        outcome = result["outcome"]
        exec_status = "executed"
    except Exception as e:
        log.error("  MCP execution failed: %s", e)
        return {
            "execution_status": f"failed: {e}",
            "simulated_outcome": {},
            "quality_delta": 0.0,
        }

    # ── Continuous Learning: Compare vs baseline ────────────────────
    baseline = state.historical_baseline
    baseline_weight = baseline.get("pred_Tablet_Weight", 200.0)
    baseline_power = baseline.get("pred_Power_Consumption_kW", 20.0)
    new_weight = outcome.get("Tablet_Weight", 200.0)
    new_power = outcome.get("Power_Consumption_kW", 20.0)

    # Yield-to-energy ratio (higher is better)
    baseline_ratio = baseline_weight / (baseline_power + 1e-8)
    new_ratio = new_weight / (new_power + 1e-8)
    quality_delta = new_ratio - baseline_ratio

    log.info("  Baseline yield/energy ratio: %.4f", baseline_ratio)
    log.info("  New      yield/energy ratio: %.4f", new_ratio)
    log.info("  Delta: %+.4f", quality_delta)

    qdrant_updated = False
    if quality_delta > 0:
        # This batch outperformed the baseline -- save as new standard!
        log.info("  IMPROVEMENT detected! Updating Qdrant vector memory...")
        _vector_memory.upsert_new_signature(
            features=state.current_telemetry,
            settings=state.proposed_settings,
            outcome=outcome,
        )
        qdrant_updated = True
    else:
        log.info("  No improvement over baseline. Qdrant NOT updated.")

    # ── FEATURE 13: Drift Detection (uses predict_with_uncertainty) ──
    global _drift_counter
    retraining_alert = False
    prediction_intervals = {}
    if _surrogate_model and _surrogate_model._is_fitted:
        try:
            # Build input feature vector matching surrogate's feature_names
            feat_vals = []
            for fname in _surrogate_model.feature_names:
                if fname in state.proposed_settings:
                    feat_vals.append(float(state.proposed_settings[fname]))
                elif fname in state.current_telemetry:
                    feat_vals.append(float(state.current_telemetry[fname]))
                else:
                    feat_vals.append(0.0)
            X_full = np.array([feat_vals], dtype=np.float64)

            uq_result = _surrogate_model.predict_with_uncertainty(X_full)
            lower = uq_result["lower"][0]
            upper = uq_result["upper"][0]
            mean = uq_result["mean"][0]

            # Build human-readable prediction intervals
            for i, target in enumerate(_surrogate_model.target_names):
                prediction_intervals[target] = {
                    "predicted": float(mean[i]),
                    "lower_10": float(lower[i]),
                    "upper_90": float(upper[i]),
                    "band_width": float(upper[i] - lower[i]),
                }

            # Check if actual outcomes are within predicted intervals
            out_of_bounds = False
            for target in _surrogate_model.target_names:
                actual_val = outcome.get(target)
                if actual_val is not None and target in prediction_intervals:
                    pi = prediction_intervals[target]
                    if actual_val < pi["lower_10"] or actual_val > pi["upper_90"]:
                        out_of_bounds = True
                        log.warning("  🚨 DRIFT: Actual %s (%.2f) outside [%.2f, %.2f]",
                                    target, actual_val, pi["lower_10"], pi["upper_90"])

            if out_of_bounds:
                _drift_counter += 1
                log.warning("  Consecutive OOB batches: %d", _drift_counter)
            else:
                _drift_counter = 0
                log.info("  ✓ All outcomes within predicted intervals")

            if _drift_counter >= 3:
                retraining_alert = True
                log.warning("  🔴 RETRAINING ALERT: %d consecutive batches outside intervals",
                            _drift_counter)
        except Exception as e:
            log.error("  Drift detection failed: %s", e)

    # ── NEW: Carbon Emission Tracking ─────────────────────────────
    power_kw = outcome.get("Power_Consumption_kW", 20.0)
    carbon_result = _carbon_tracker.track_batch(
        batch_id=state.batch_id,
        power_kw=power_kw,
    )
    log.info("  Carbon: %.3f kgCO₂ (cumulative: %.3f kgCO₂)",
             carbon_result["carbon_kg"], carbon_result["cumulative_carbon_kg"])

    # ── NEW: Log decision to decision memory ─────────────────────
    _decision_memory.log_decision(
        batch_id=state.batch_id,
        proposed_settings=state.proposed_settings,
        approved=True,
        feedback=state.human_feedback,
        quality_delta=float(quality_delta),
    )

    # ── NEW: Add to batch history ─────────────────────────────────
    _batch_history.add_batch(
        batch_id=state.batch_id,
        proposed_settings=state.proposed_settings,
        simulated_outcome=outcome,
        quality_delta=float(quality_delta),
        qdrant_updated=qdrant_updated,
        human_approved=True,
        human_feedback=state.human_feedback,
        carbon_metrics=carbon_result,
        energy_anomalies=state.energy_anomalies,
    )

    return _to_native({
        "execution_status": exec_status,
        "simulated_outcome": outcome,
        "quality_delta": float(quality_delta),
        "qdrant_updated": qdrant_updated,
        "carbon_metrics": carbon_result,
        "retraining_alert": retraining_alert,
        "prediction_intervals": prediction_intervals,
    })


# =====================================================================
# 6. ROUTING LOGIC
# =====================================================================
def should_execute(state: ManufacturingState) -> str:
    """Conditional edge: route to execution or end based on approval."""
    if state.human_approved:
        return "execution_node"
    return END


# =====================================================================
# 7. GRAPH BUILDER
# =====================================================================
def build_orchestration_graph() -> StateGraph:
    """Construct the LangGraph state machine for manufacturing
    optimization orchestration.

    Graph topology:
      START -> data_router -> proxy_caller -> hitl_gate -> [execution | END]
                                                               |
                                                               v
                                                              END
    """
    graph = StateGraph(ManufacturingState)

    # ── Add nodes ───────────────────────────────────────────────────
    graph.add_node("data_router", data_router_node)
    graph.add_node("proxy_caller", proxy_caller_node)
    graph.add_node("hitl_gate", hitl_gate_node)
    graph.add_node("execution_node", execution_node)

    # ── Add edges ───────────────────────────────────────────────────
    graph.add_edge(START, "data_router")
    graph.add_edge("data_router", "proxy_caller")
    graph.add_edge("proxy_caller", "hitl_gate")
    graph.add_conditional_edges("hitl_gate", should_execute)
    graph.add_edge("execution_node", END)

    return graph


def compile_graph():
    """Build and compile the graph with memory checkpointing for HITL."""
    graph = build_orchestration_graph()
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    return compiled, checkpointer


# =====================================================================
# 8. INITIALIZATION  --  Bootstrap all components
# =====================================================================
def initialize_system(
    golden_signatures_path: Optional[str] = None,
    max_signatures: int = 200,
) -> None:
    """Bootstrap all orchestration components:
      - Load and train the PyTorch proxy on Golden Signatures
      - Initialize Qdrant vector memory and ingest signatures
      - Set up MCP executor and Openlayer monitor

    Parameters
    ----------
    golden_signatures_path : str, optional
        Path to golden_signatures.csv. Defaults to test-data/ dir.
    max_signatures : int
        Max signatures to ingest (for fast demo).
    """
    global _vector_memory, _mcp_executor, _openlayer
    global _proxy_model, _input_scaler, _output_scaler
    global _carbon_tracker, _batch_history, _energy_analyzer
    global _decision_memory, _surrogate_model

    if golden_signatures_path is None:
        golden_signatures_path = os.path.join(DATA_DIR, "golden_signatures.csv")

    # ── Load Golden Signatures ──────────────────────────────────────
    log.info("Loading Golden Signatures from: %s", golden_signatures_path)
    golden_df = pd.read_csv(golden_signatures_path)
    golden_df = golden_df.head(max_signatures)  # limit for demo speed
    log.info("  Loaded %d signatures", len(golden_df))

    # ── Train PyTorch Proxy ─────────────────────────────────────────
    context_cols = [c for c in golden_df.columns if c.startswith("ctx_")]
    decision_cols = [c for c in DECISION_VARS if c in golden_df.columns]

    X = golden_df[context_cols].values.astype(np.float32)
    y = golden_df[decision_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

    _input_scaler = StandardScaler()
    _output_scaler = StandardScaler()
    X_scaled = _input_scaler.fit_transform(X).astype(np.float32)
    y_scaled = _output_scaler.fit_transform(y).astype(np.float32)

    _proxy_model = OptimizationProxy(
        input_dim=X_scaled.shape[1],
        output_dim=len(decision_cols),
        hidden_dims=(256, 128, 64),
        dropout=0.15,
    ).to(DEVICE)

    # Quick training (for demo speed)
    optimizer = torch.optim.AdamW(_proxy_model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    X_tensor = torch.tensor(X_scaled).to(DEVICE)
    y_tensor = torch.tensor(y_scaled).to(DEVICE)

    _proxy_model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        pred = _proxy_model.backbone(X_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0:
            log.info("  Proxy training epoch %d: loss=%.6f", epoch, loss.item())

    log.info("  Proxy training complete")

    # ── Initialize Qdrant with same context columns ─────────────────
    _vector_memory = VectorMemory(context_cols=context_cols)
    _vector_memory.ingest_golden_signatures(golden_df)

    # ── Initialize MCP executor & Openlayer ─────────────────────────
    _mcp_executor = MCPToolExecutor()
    _openlayer = OpenlayerMonitor()

    # ── NEW: Initialize feature modules ─────────────────────────────
    _carbon_tracker = CarbonTracker()
    _batch_history = BatchHistoryStore()
    _energy_analyzer = EnergyPatternAnalyzer()
    _decision_memory = DecisionMemory()

    # ── NEW: Train surrogate and keep reference for feature importances
    _surrogate_model = SurrogateModel()
    try:
        from data_layer import build_training_dataset
        training_data = build_training_dataset()
        _surrogate_model.fit(training_data)
        log.info("  Surrogate model trained for feature importances")
    except Exception as e:
        log.warning("  Could not train surrogate for SHAP: %s", e)
        _surrogate_model = None

    log.info("System initialization complete!")


# =====================================================================
# 9. CLI ENTRY POINT  --  Full demo flow
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  ORCHESTRATION LAYER  --  Phase 2: Agentic Orchestration")
    print("  LangGraph + Qdrant + MCP + HITL Workflow")
    print("=" * 60)
    print()

    # ── Initialize all subsystems ───────────────────────────────────
    initialize_system(max_signatures=100)

    # ── Build the LangGraph ─────────────────────────────────────────
    compiled_graph, checkpointer = compile_graph()
    print("\nGraph compiled successfully!")
    print(f"  Nodes: data_router -> proxy_caller -> hitl_gate -> execution_node")
    print(f"  HITL interrupt at: hitl_gate (uses interrupt() + Command resume)")

    # ── Simulate incoming telemetry ─────────────────────────────────
    # Use ACTUAL context column names from Golden Signatures so the
    # raw numerical vector query matches the Qdrant collection schema.
    golden_for_sim = pd.read_csv(
        os.path.join(DATA_DIR, "golden_signatures.csv"), nrows=1
    )
    sim_context_cols = [c for c in golden_for_sim.columns if c.startswith("ctx_")]
    # Take the first row as simulated "live" telemetry with slight perturbation
    rng = np.random.default_rng(42)
    simulated_telemetry = {
        c: float(golden_for_sim[c].iloc[0]) * (1 + rng.normal(0, 0.05))
        for c in sim_context_cols
    }

    initial_state = ManufacturingState(
        batch_id="LIVE-001",
        current_telemetry=simulated_telemetry,
    )

    thread_config = {"configurable": {"thread_id": "demo-thread-001"}}

    # ── Run graph (will pause at HITL gate) ─────────────────────────
    print("\n" + "-" * 60)
    print("Running graph... (will pause at HITL gate)")
    print("-" * 60)

    result = None
    for event in compiled_graph.stream(
        initial_state.model_dump(), thread_config, stream_mode="values"
    ):
        result = event

    # ── Check if interrupted ────────────────────────────────────────
    snapshot = compiled_graph.get_state(thread_config)

    if snapshot.next:
        print("\n" + "=" * 60)
        print("HITL INTERRUPT: Graph paused at:", snapshot.next)
        print("  Proposed settings are awaiting human approval.")
        print()
        print("  In production, the React dashboard would call:")
        print("    POST /api/approve  {'approved': true}")
        print()
        print("  Simulating APPROVAL for demo...")
        print("=" * 60)

        # ── Resume with simulated human approval ────────────────────
        for event in compiled_graph.stream(
            Command(resume={"approved": True, "feedback": "Looks good!"}),
            thread_config,
            stream_mode="values",
        ):
            result = event

    # ── Final state ─────────────────────────────────────────────────
    if result:
        print("\n" + "=" * 60)
        print("FINAL ORCHESTRATION STATE")
        print("=" * 60)
        print(f"  Batch ID        : {result.get('batch_id', 'N/A')}")
        print(f"  Execution Status: {result.get('execution_status', 'N/A')}")
        print(f"  Human Approved  : {result.get('human_approved', 'N/A')}")
        print(f"  Quality Delta   : {result.get('quality_delta', 0.0):+.4f}")
        print(f"  Qdrant Updated  : {result.get('qdrant_updated', False)}")

        outcome = result.get("simulated_outcome", {})
        if outcome:
            print(f"\n  Simulated Production Outcome:")
            for k, v in outcome.items():
                print(f"    {k}: {v}")

        settings = result.get("proposed_settings", {})
        if settings:
            print(f"\n  Applied Machine Settings:")
            for k, v in settings.items():
                lo, hi = DECISION_BOUNDS.get(k, (0, 999))
                print(f"    {k}: {v:.2f}  [{lo}, {hi}]")

    print("\n" + "=" * 60)
    print("  Phase 2: Agentic Orchestration COMPLETE")
    print("=" * 60)
