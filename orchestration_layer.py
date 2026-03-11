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

# ── Sentence Transformers (Embedding) ──────────────────────────────────
from sentence_transformers import SentenceTransformer

# ── Phase 1 Imports ────────────────────────────────────────────────────
from data_layer import build_training_dataset, DATA_DIR, SENSOR_COLS
from offline_optimizer import DECISION_VARS, DECISION_BOUNDS, TARGET_COLS
from model_layer import (
    OptimizationProxy,
    RepairLayer,
    InferenceEngine,
    DEVICE,
)

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
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"   # fast, 384-d embeddings
EMBEDDING_DIM     = 384

# Simulated physical constraints for the MCP tool
MCP_TOOL_NAME = "execute_machine_parameters"


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for
    JSON/msgpack serialization compatibility with LangGraph checkpoints."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
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


# =====================================================================
# 2. VECTOR MEMORY  --  Qdrant Manager
# =====================================================================
class VectorMemory:
    """Manages the Qdrant in-memory vector database for Golden Signature
    retrieval and continuous learning updates.

    Uses SentenceTransformer to embed numeric feature vectors into a
    semantic space for approximate nearest-neighbor search.
    """

    def __init__(self) -> None:
        log.info("Initializing Qdrant in-memory vector database...")
        self.client = QdrantClient(":memory:")  # no external server needed
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # Create collection
        self.client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        log.info(
            "Qdrant collection '%s' created (dim=%d, cosine)",
            QDRANT_COLLECTION, EMBEDDING_DIM,
        )

    def _feature_to_text(self, features: Dict[str, float]) -> str:
        """Convert a numeric feature dict to a text string for embedding."""
        parts = [f"{k}={v:.4f}" for k, v in sorted(features.items())]
        return " ".join(parts)

    def _embed(self, text: str) -> List[float]:
        """Embed a text string using SentenceTransformer."""
        vec = self.embedder.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def ingest_golden_signatures(self, golden_df: pd.DataFrame) -> int:
        """Bulk-load Golden Signatures into Qdrant.

        Parameters
        ----------
        golden_df : pd.DataFrame
            Golden Signatures CSV from offline_optimizer.

        Returns
        -------
        int
            Number of points ingested.
        """
        points: List[PointStruct] = []
        decision_cols = [c for c in DECISION_VARS if c in golden_df.columns]
        context_cols = [c for c in golden_df.columns if c.startswith("ctx_")]
        pred_cols = [c for c in golden_df.columns if c.startswith("pred_")]

        for idx, row in golden_df.iterrows():
            # Build feature text from context columns for embedding
            feature_dict = {c: float(row[c]) for c in context_cols[:20]}
            text = self._feature_to_text(feature_dict)
            vector = self._embed(text)

            # Payload = decision vars + predictions
            payload: Dict[str, Any] = {}
            for c in decision_cols:
                payload[c] = float(row[c])
            for c in pred_cols:
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
            "Ingested %d Golden Signatures into Qdrant '%s'",
            len(points), QDRANT_COLLECTION,
        )
        return len(points)

    def query_nearest(
        self, telemetry: Dict[str, float], top_k: int = 1
    ) -> Dict[str, Any]:
        """Find the closest historical Golden Signature to incoming telemetry.

        Parameters
        ----------
        telemetry : Dict[str, float]
            Current telemetry feature dict.
        top_k : int
            Number of nearest neighbors to retrieve.

        Returns
        -------
        Dict[str, Any]
            Best matching Golden Signature payload + similarity score.
        """
        text = self._feature_to_text(telemetry)
        vector = self._embed(text)

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
        self, features: Dict[str, float], settings: Dict[str, float],
        outcome: Dict[str, float],
    ) -> int:
        """Add a new learned Golden Signature to Qdrant (continuous learning).

        Parameters
        ----------
        features : Dict[str, float]
            Context features that led to the good outcome.
        settings : Dict[str, float]
            The machine settings that were applied.
        outcome : Dict[str, float]
            The simulated/actual production outcome.

        Returns
        -------
        int
            The point ID of the new entry.
        """
        text = self._feature_to_text(features)
        vector = self._embed(text)

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

    return _to_native({
        "historical_baseline": baseline,
        "baseline_score": score,
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

    return _to_native({
        "proposed_settings": proposed,
        "raw_settings": raw_settings,
    })


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

    # ── INTERRUPT: Pause graph execution here ───────────────────────
    # The interrupt() call pauses the graph and surfaces the payload
    # to the caller. The graph will resume when the caller invokes
    # graph.invoke(Command(resume={"approved": True/False}), config)
    human_decision = interrupt({
        "message": "Please review and approve the proposed machine settings.",
        "batch_id": state.batch_id,
        "proposed_settings": state.proposed_settings,
        "baseline_score": state.baseline_score,
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
        }
    else:
        log.info("  REJECTED by human operator. Feedback: %s", feedback)
        return {
            "human_approved": False,
            "human_feedback": feedback,
            "execution_status": "rejected",
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

    return _to_native({
        "execution_status": exec_status,
        "simulated_outcome": outcome,
        "quality_delta": float(quality_delta),
        "qdrant_updated": qdrant_updated,
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

    # ── Initialize Qdrant ───────────────────────────────────────────
    _vector_memory = VectorMemory()
    _vector_memory.ingest_golden_signatures(golden_df)

    # ── Initialize MCP executor & Openlayer ─────────────────────────
    _mcp_executor = MCPToolExecutor()
    _openlayer = OpenlayerMonitor()

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
    simulated_telemetry = {
        "Temperature_C": 45.2,
        "Pressure_Bar": 3.5,
        "Humidity_Percent": 62.1,
        "Motor_Speed_RPM": 1200.0,
        "Compression_Force_kN": 15.3,
        "Flow_Rate_LPM": 8.7,
        "Power_Consumption_kW": 22.5,
        "Vibration_mm_s": 3.1,
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
