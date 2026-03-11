# Intelli-Credit-Aveva — Industrial AI Optimization Engine

> **Hackathon Prototype** — IITH x Aveva  
> Balancing conflicting manufacturing targets: **maximizing yield/quality** while **minimizing energy consumption** using a 5-layer modular AI architecture.

---

## Architecture Overview

```
+==================================================================+
|                    PHASE 1: THE CORE ENGINE                       |
|                                                                   |
|  +-------------+    +------------------+    +------------------+  |
|  | DATA LAYER  |--->| OFFLINE OPTIMIZER |--->|   MODEL LAYER    |  |
|  |             |    |                  |    |                  |  |
|  | Ingestion   |    | XGBoost Surrogate|    | PyTorch Proxy    |  |
|  | Validation  |    | NSGA-II Pareto   |    | + Repair Layer   |  |
|  | Feature Eng |    | Golden Signatures|    | (< 1ms inference)|  |
|  +-------------+    +------------------+    +------------------+  |
+==================================================================+
                              |
                              v
+==================================================================+
|                PHASE 2: AGENTIC ORCHESTRATION                     |
|                                                                   |
|  +------------+    +--------------+    +------+    +-----------+  |
|  | Data       |--->| Proxy Caller |--->| HITL |--->| Execution |  |
|  | Router     |    |   (Brain)    |    | Gate |    |   Node    |  |
|  +-----+------+    +------+-------+    +------+    +-----+-----+  |
|        |                  |                              |        |
|        v                  v                              v        |
|    [Qdrant]        [LangSmith +              [MCP Tool + Qdrant   |
|     Vector          Openlayer                 Continuous          |
|     Memory          Tracing]                  Learning]           |
+==================================================================+
```

---

## Phase 1: The Core Engine

### Layer 1 — Data Layer (`data_layer.py`)
- **Ingestion & Validation**: Loads `.xlsx` telemetry + production summaries with automated null handling, IQR-based anomaly detection, and physical sensor range checks.
- **Phase-Aware Feature Engineering**: Slices telemetry by 8 manufacturing phases. Extracts per-phase:
  - Statistical aggregates (mean, std, min, max)
  - **Thermal Ramp Rate** — first-order derivative of Temperature over Time
  - **Power AUC** — trapezoidal integral of Power Consumption
  - **Vibration AUC** — trapezoidal integral of Vibration
- **Gaussian-Noise Data Augmentation**: Synthesizes varied telemetry features (sigma = 8%) correlated with production settings to prevent **feature collapse** from single-batch telemetry.
- **Output**: 60-row x 295-column merged training dataset.

### Layer 2 — Offline Optimizer (`offline_optimizer.py`)
- **XGBoost Surrogate Model**: Multi-output regressor mapping 7 machine settings + 280 telemetry features to quality targets + energy.
- **NSGA-II Genetic Algorithm** (custom implementation):
  - Simulated Binary Crossover (SBX, eta=20), Polynomial Mutation (eta=20)
  - Constraint-aware Non-Dominated Sorting + Crowding Distance
- **Objectives**: Maximize `Tablet_Weight`, Minimize `Power_Consumption_kW`
- **Constraints**: `Friability in [0.1, 1.0]`, `Hardness in [4, 10]`
- **Output**: 4,000 Pareto-optimal "Golden Signatures" across 20 historical contexts.

### Layer 3 — Model Layer (`model_layer.py`)
- **Optimization Proxy**: 115K-parameter feed-forward neural network (BatchNorm, GELU, Dropout x3) trained on Golden Signatures.
- **The Repair Layer** (Critical Innovation):
  - **Box Constraints**: `torch.clamp()` projects outputs into physically feasible bounds
  - **Coupling Constraints**: Alternating projections enforce interactive physics (e.g., high compression force de-rates max speed)
  - Fully differentiable — gradients flow through during backpropagation
- **Inference**: < 1ms per batch, 100% constraint adherence verified.

---

## Phase 2: Agentic Orchestration (`orchestration_layer.py`)

A **LangGraph state machine** that orchestrates the full optimization lifecycle:

### Node 1 — Data Router & Memory Retrieval
- Receives simulated live telemetry
- Queries **Qdrant** in-memory vector DB (SentenceTransformer embeddings) for the nearest matching Golden Signature baseline

### Node 2 — Proxy Caller (The Brain)
- Feeds context features into the Phase 1 PyTorch Optimization Proxy
- Retrieves instantly generated, physically repaired settings (< 1ms)
- **LangSmith `@traceable`** decorator for observability
- **Openlayer** callback handler monitors for hallucinated parameters

### Conditional Edge — HITL Gate
- Uses LangGraph `interrupt()` to **pause execution** and surface proposed settings for human review
- Resumes via `Command(resume={"approved": True/False})` — ready for the Phase 3 React dashboard
- API hooks documented at `POST /api/approve` and `GET /api/pending-approval`

### Node 3 — Execution & Continuous Learning
- **MCP Tool** (`execute_machine_parameters`) simulates secure factory execution via `langchain-mcp-adapters`
- Compares new outcome against historical baseline (yield-to-energy ratio)
- If improved, **upserts new signature into Qdrant** as the new golden standard

### Pydantic State
Strictly typed via `ManufacturingState(BaseModel)`:
`batch_id`, `current_telemetry`, `historical_baseline`, `proposed_settings`, `human_approved`, `execution_status`, `simulated_outcome`, `quality_delta`, `qdrant_updated`

---

## Data Assets

| File | Description | Shape |
|------|-------------|-------|
| `test-data/_h_batch_process_data.xlsx` | Time-series telemetry (8 phases) | 211 x 11 |
| `test-data/_h_batch_production_data.xlsx` | Batch summary (quality + settings) | 60 x 15 |
| `test-data/training_dataset.csv` | Merged + augmented training data | 60 x 295 |
| `test-data/golden_signatures.csv` | NSGA-II Pareto-optimal solutions | 4,000 x 297 |

---

## Quick Start

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost torch openpyxl
pip install langgraph qdrant-client langchain-mcp-adapters openlayer sentence-transformers

# Phase 1: Core Engine
python -X utf8 data_layer.py            # Data ingestion + augmentation
python -X utf8 offline_optimizer.py     # XGBoost surrogate + NSGA-II
python -X utf8 model_layer.py           # PyTorch proxy + repair layer

# Phase 2: Agentic Orchestration
python -X utf8 orchestration_layer.py   # LangGraph + Qdrant + HITL workflow
```

> **Note**: Use `python -X utf8` on Windows to avoid encoding issues with console output.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Data Engineering | Pandas, NumPy, Scikit-learn |
| Surrogate Model | XGBoost (Multi-output Regressor) |
| Optimization | NSGA-II (custom implementation) |
| Neural Proxy | PyTorch (AdamW, CosineAnnealing) |
| Constraint Enforcement | Deterministic Repair Layer (`torch.clamp` + alternating projections) |
| Orchestration | LangGraph (Pydantic state machine) |
| Vector Memory | Qdrant (in-memory, SentenceTransformer embeddings) |
| Secure Execution | Model Context Protocol (langchain-mcp-adapters) |
| Observability | LangSmith (`@traceable`) + Openlayer |
| HITL Workflow | LangGraph `interrupt()` + `Command(resume=...)` |

---

## License

See [LICENSE](LICENSE) for details.