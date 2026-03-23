import os
import random
import uuid
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

# Phase 2 imports
from orchestration_layer import (
    initialize_system,
    compile_graph,
    ManufacturingState,
    DECISION_BOUNDS,
    _carbon_tracker,
    _batch_history,
    _decision_memory,
    _surrogate_model,
    _current_priorities,
    _regulatory_targets,
    _audit_ledger,
)
from langgraph.types import Command

app = FastAPI(title="Industrial AI Optimizer API")

# Allow local Vite server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold graph instances
compiled_graph = None
checkpointer = None
latest_batch_id = None

class TelemetryPayload(BaseModel):
    batch_id: str
    telemetry: Dict[str, float]

class DecisionPayload(BaseModel):
    batch_id: str
    approved: bool
    feedback: Optional[str] = ""

class PriorityPayload(BaseModel):
    batch_id: Optional[str] = None
    priority_value: float
    priority_type: str = "yield_vs_energy"
    objective_primary: str = "Tablet_Weight"
    objective_secondary: str = "Power_Consumption_kW"

class RegulatoryPayload(BaseModel):
    max_carbon_per_batch_kg: Optional[float] = None
    max_power_per_batch_kwh: Optional[float] = None
    min_yield_pct: Optional[float] = None
    min_hardness: Optional[float] = None
    max_friability: Optional[float] = None
    emission_factor_name: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    global compiled_graph, checkpointer
    print("🚀 Initializing backend subsystems (PyTorch Proxy, Qdrant)...")
    initialize_system(max_signatures=100)
    compiled_graph, checkpointer = compile_graph()
    print("✅ Backend initialized and graph compiled.")

def run_graph_background(batch_id: str, telemetry: Dict[str, float]):
    """Runs the graph in a background task so the API doesn't block while the proxy runs."""
    config = {"configurable": {"thread_id": batch_id}}
    initial_state = ManufacturingState(
        batch_id=batch_id,
        current_telemetry=telemetry
    )
    
    # Run the graph; it will pause at the HITL interrupt
    for _ in compiled_graph.stream(initial_state.model_dump(), config, stream_mode="values"):
        pass

@app.post("/api/trigger_batch")
async def trigger_batch(payload: TelemetryPayload, background_tasks: BackgroundTasks):
    """Initiates the LangGraph optimization workflow."""
    global latest_batch_id
    latest_batch_id = payload.batch_id
    background_tasks.add_task(run_graph_background, payload.batch_id, payload.telemetry)
    return {"status": "started", "batch_id": payload.batch_id}

@app.get("/api/graph_state")
async def get_graph_state(batch_id: str):
    """Polls the current execution state of the graph."""
    global latest_batch_id
    if batch_id == "LATEST_KNOWN":
        if not latest_batch_id:
            return {"status": "not_found", "message": "No active batch"}
        batch_id = latest_batch_id

    config = {"configurable": {"thread_id": batch_id}}
    state_snapshot = compiled_graph.get_state(config)
    
    if not state_snapshot:
        return {"status": "not_found", "message": f"No state found for batch {batch_id}"}
        
    state_vals = state_snapshot.values
    paused = bool(state_snapshot.next)
    
    # Build a clean response for the frontend
    return {
        "status": "active" if state_vals.get("execution_status") == "pending" else state_vals.get("execution_status"),
        "paused_for_hitl": paused,
        "batch_id": state_vals.get("batch_id"),
        "current_telemetry": state_vals.get("current_telemetry", {}),
        "historical_baseline": state_vals.get("historical_baseline", {}),
        "proposed_settings": state_vals.get("proposed_settings", {}),
        "raw_settings": state_vals.get("raw_settings", {}),
        "simulated_outcome": state_vals.get("simulated_outcome", {}),
        "quality_delta": state_vals.get("quality_delta", 0.0),
        "qdrant_updated": state_vals.get("qdrant_updated", False),
        "baseline_score": state_vals.get("baseline_score", 0.0),
        "bounds": DECISION_BOUNDS,
        # NEW fields
        "carbon_metrics": state_vals.get("carbon_metrics", {}),
        "energy_anomalies": state_vals.get("energy_anomalies", []),
        "asset_health_score": state_vals.get("asset_health_score", 100.0),
        "energy_recommendations": state_vals.get("energy_recommendations", []),
        "past_decision_warnings": state_vals.get("past_decision_warnings", []),
        "optimization_priorities": state_vals.get("optimization_priorities", _current_priorities),
        # Phase 2 fields
        "no_confident_match": state_vals.get("no_confident_match", False),
        "novelty_warning": state_vals.get("novelty_warning", {}),
        "prediction_intervals": state_vals.get("prediction_intervals", {}),
        "retraining_alert": state_vals.get("retraining_alert", False),
    }

@app.post("/api/execute_decision")
async def execute_decision(payload: DecisionPayload):
    """Resumes the graph with the human's decision."""
    config = {"configurable": {"thread_id": payload.batch_id}}
    state_snapshot = compiled_graph.get_state(config)
    
    if not state_snapshot or not state_snapshot.next:
        raise HTTPException(status_code=400, detail="Graph is not currently paused awaiting HITL.")
        
    # Resume the graph
    print(f"👉 Received HITL decision for {payload.batch_id}: approved={payload.approved}")
    for _ in compiled_graph.stream(
        Command(resume={"approved": payload.approved, "feedback": payload.feedback}),
        config,
        stream_mode="values"
    ):
        pass
        
    return {"status": "resumed", "approved": payload.approved}

@app.post("/api/new_batch")
async def new_batch(background_tasks: BackgroundTasks):
    """Generate a new batch with realistic telemetry and trigger the optimization graph.
    
    Samples a real Golden Signature row from the CSV and adds slight perturbation
    so the Qdrant vector search produces meaningful (high) match scores.
    """
    global latest_batch_id
    import pandas as pd
    import numpy as np
    from data_layer import DATA_DIR
    import os

    batch_id = f"BATCH-{uuid.uuid4().hex[:6].upper()}"
    latest_batch_id = batch_id
    
    # Load a random Golden Signature row to use as realistic telemetry
    golden_path = os.path.join(DATA_DIR, "golden_signatures.csv")
    try:
        golden_df = pd.read_csv(golden_path, nrows=200)  # match initialize_system max_signatures
        ctx_cols = [c for c in golden_df.columns if c.startswith("ctx_")]
        
        # Pick a random row from the SAME pool that Qdrant has ingested
        row = golden_df.sample(1).iloc[0]
        rng = np.random.default_rng()
        
        # Build telemetry with the CORRECT ctx_ column names + slight noise
        mock_telemetry = {}
        # 1 in 3 chance of simulating a "drifting" batch to show off Novelty Warning
        is_anomalous = rng.random() > 0.65
        
        for col in ctx_cols:
            base_val = float(row[col]) if pd.notna(row[col]) else 0.0
            
            if is_anomalous and rng.random() > 0.5:
                # To break Cosine Similarity across 284 dims, we must significantly
                # alter the vector's direction by zeroing out or flipping signs of many features
                noise = -base_val * rng.uniform(0.5, 1.5)
            else:
                # Standard small 1-3% noise
                noise = rng.normal(0, 0.02) * abs(base_val) if abs(base_val) > 1e-6 else rng.normal(0, 0.01)
                
            mock_telemetry[col] = round(base_val + noise, 4)
        
        # Also include human-readable sensor values for the Dashboard UI
        mock_telemetry["Temperature_C"] = mock_telemetry.get("ctx_Preparation_Temperature_C_mean", round(random.uniform(60, 95), 1))
        mock_telemetry["Pressure_Bar"] = mock_telemetry.get("ctx_Preparation_Pressure_Bar_mean", round(random.uniform(1.0, 3.0), 2))
        mock_telemetry["Humidity_Percent"] = mock_telemetry.get("ctx_Preparation_Humidity_Percent_mean", round(random.uniform(25, 60), 1))
        mock_telemetry["Motor_Speed_RPM"] = mock_telemetry.get("ctx_Compression_Motor_Speed_RPM_mean", round(random.uniform(40, 100), 1))
        mock_telemetry["Compression_Force_kN"] = mock_telemetry.get("ctx_Compression_Compression_Force_kN_mean", round(random.uniform(8, 25), 1))
        mock_telemetry["Flow_Rate_LPM"] = mock_telemetry.get("ctx_Granulation_Flow_Rate_LPM_mean", round(random.uniform(80, 200), 1))
        mock_telemetry["Power_Consumption_kW"] = mock_telemetry.get("ctx_Compression_Power_Consumption_kW_mean", round(random.uniform(15, 50), 1))
        mock_telemetry["Vibration_mm_s"] = mock_telemetry.get("ctx_Compression_Vibration_mm_s_mean", round(random.uniform(2, 8), 2))
        
    except Exception as e:
        print(f"[WARNING] Could not load golden signatures for telemetry: {e}")
        # Fallback to basic random telemetry
        mock_telemetry = {
            "Temperature_C": round(random.uniform(60, 95), 1),
            "Pressure_Bar": round(random.uniform(1.0, 3.0), 2),
            "Humidity_Percent": round(random.uniform(25, 60), 1),
            "Motor_Speed_RPM": round(random.uniform(40, 100), 1),
            "Power_Consumption_kW": round(random.uniform(15, 50), 1),
            "Vibration_mm_s": round(random.uniform(2, 8), 2),
        }
    
    background_tasks.add_task(run_graph_background, batch_id, mock_telemetry)
    print(f"🆕 New batch triggered from dashboard: {batch_id}")
    return {"status": "started", "batch_id": batch_id}



# =====================================================================
# NEW ENDPOINTS
# =====================================================================

@app.get("/api/priority_modes")
async def get_priority_modes():
    """Get available priority balancing modes."""
    from orchestration_layer import PRIORITY_MODES, _current_priorities
    return {
        "modes": PRIORITY_MODES,
        "current": _current_priorities,
    }


@app.post("/api/update_priorities")
async def update_priorities(payload: PriorityPayload):
    """Update optimization priorities from the frontend slider / dropdown."""
    from orchestration_layer import _current_priorities as priorities, PRIORITY_MODES
    priorities["priority_value"] = payload.priority_value
    priorities["mode"] = payload.priority_type

    # Auto-set objectives based on mode
    mode_config = PRIORITY_MODES.get(payload.priority_type, {})
    if mode_config:
        priorities["objective_primary"] = mode_config.get("primary", priorities["objective_primary"])
        priorities["objective_secondary"] = mode_config.get("secondary", priorities["objective_secondary"])

    if payload.objective_primary:
        priorities["objective_primary"] = payload.objective_primary
    if payload.objective_secondary:
        priorities["objective_secondary"] = payload.objective_secondary

    print(f"🎚️ Priority updated: {payload.priority_type} = {payload.priority_value} "
          f"({priorities['objective_primary']} vs {priorities['objective_secondary']})")
    return {
        "status": "success",
        "priorities": priorities,
    }


@app.get("/api/carbon_metrics")
async def get_carbon_metrics():
    """Get cumulative and per-batch carbon emissions."""
    from orchestration_layer import _carbon_tracker
    if _carbon_tracker is None:
        return {"error": "Carbon tracker not initialized"}
    return _carbon_tracker.get_summary()


@app.get("/api/batch_history")
async def get_batch_history():
    """Get all historical batch records."""
    from orchestration_layer import _batch_history
    if _batch_history is None:
        return {"records": [], "stats": {}}
    return {
        "records": _batch_history.get_all(),
        "stats": _batch_history.get_summary_stats(),
    }




class ChatPayload(BaseModel):
    """Payload for LLM-based operator chat."""
    message: str
    batch_id: Optional[str] = None


@app.post("/api/chat")
async def operator_chat(payload: ChatPayload):
    """LLM-powered natural language interface for the operator.
    
    The operator types a message like 'Looks good, approve it' or
    'What if we increase the speed?'. Gemini interprets the intent
    and returns a structured response with action + explanation.
    """
    from gemini_llm import process_operator_message
    
    bid = payload.batch_id or latest_batch_id
    state_data = {}
    if bid:
        config = {"configurable": {"thread_id": bid}}
        state_snapshot = compiled_graph.get_state(config)
        if state_snapshot:
            state_data = state_snapshot.values

    result = process_operator_message(payload.message, state_data)
    
    # If the LLM determined this is an approve/reject, auto-execute the HITL decision
    if result.get("approved") is not None and bid:
        config = {"configurable": {"thread_id": bid}}
        state_snapshot = compiled_graph.get_state(config)
        if state_snapshot and state_snapshot.next:
            try:
                for _ in compiled_graph.stream(
                    Command(resume={"approved": result["approved"], "feedback": result.get("feedback", payload.message)}),
                    config,
                    stream_mode="values"
                ):
                    pass
                result["execution_resumed"] = True
                
                # If approved and execution complete, get outcome explanation
                if result["approved"]:
                    from gemini_llm import explain_outcome
                    new_state = compiled_graph.get_state(config)
                    if new_state:
                        sv = new_state.values
                        outcome = sv.get("simulated_outcome", {})
                        carbon = sv.get("carbon_metrics", {}).get("carbon_kg", 0.0)
                        qd = sv.get("quality_delta", 0.0)
                        explanation = explain_outcome(bid, outcome, qd, carbon)
                        result["response"] += f"\n\n{explanation}"
            except Exception as e:
                result["execution_error"] = str(e)
    
    return result


@app.post("/api/hitl_decision")
async def hitl_decision(payload: DecisionPayload):
    """Resume the graph with the human decision (button-based or from chat)."""
    config = {"configurable": {"thread_id": payload.batch_id}}
    state_snapshot = compiled_graph.get_state(config)

    if not state_snapshot or not state_snapshot.next:
        raise HTTPException(status_code=400, detail="Graph is not currently paused awaiting HITL.")

    # If rejected, also log to audit ledger
    if not payload.approved:
        from orchestration_layer import _audit_ledger, _decision_memory
        if _audit_ledger:
            _audit_ledger.append(
                batch_id=payload.batch_id,
                ai_suggestion=state_snapshot.values.get("proposed_settings", {}),
                human_decision="rejected",
                human_feedback=payload.feedback or "",
                carbon_kg=0.0,
                power_kw=0.0,
            )
        if _decision_memory:
            _decision_memory.log_decision(
                batch_id=payload.batch_id,
                proposed_settings=state_snapshot.values.get("proposed_settings", {}),
                approved=False,
                feedback=payload.feedback or "",
                quality_delta=0.0,
            )

    for _ in compiled_graph.stream(
        Command(resume={"approved": payload.approved, "feedback": payload.feedback}),
        config,
        stream_mode="values"
    ):
        pass

    return {"status": "resumed", "approved": payload.approved}




class WhatIfPayload(BaseModel):
    """Payload for Digital Twin what-if simulation."""
    settings: Dict[str, float]
    batch_id: Optional[str] = None


@app.get("/api/feature_importance")
async def get_feature_importance(top_n: int = 15):
    """Get model-level XGBoost feature importances (gain-based)."""
    from orchestration_layer import _surrogate_model
    if _surrogate_model is None or not _surrogate_model._is_fitted:
        return {"features": {}, "message": "Surrogate model not available"}
    importances = _surrogate_model.get_feature_importances(top_n=top_n)
    return {"features": importances}


@app.get("/api/shap_values")
async def get_shap_values(batch_id: Optional[str] = None):
    """Get instance-level SHAP values for the current batch prediction.

    Uses shap.TreeExplainer on each per-target XGBoost model to compute
    how each feature pushed the prediction UP or DOWN from the base value.
    """
    from orchestration_layer import _surrogate_model
    import numpy as np

    if _surrogate_model is None or not _surrogate_model._is_fitted:
        return {"targets": {}, "message": "Surrogate model not available"}

    bid = batch_id or latest_batch_id
    if not bid:
        return {"targets": {}, "message": "No active batch"}

    config = {"configurable": {"thread_id": bid}}
    state_snapshot = compiled_graph.get_state(config)
    if not state_snapshot:
        return {"targets": {}, "message": "No state found"}

    sv = state_snapshot.values
    proposed = sv.get("proposed_settings", {})
    telemetry = sv.get("current_telemetry", {})

    # Build the input vector for the surrogate model
    feature_names = _surrogate_model.feature_names
    combined = {**telemetry, **proposed}
    input_vec = []
    for f in feature_names:
        input_vec.append(combined.get(f, 0.0))
    X = np.array([input_vec])

    try:
        import shap

        result = {}
        model = _surrogate_model.model  # MultiOutputRegressor

        for i, target_name in enumerate(_surrogate_model.target_names):
            estimator = model.estimators_[i]
            explainer = shap.TreeExplainer(estimator)
            shap_vals = explainer.shap_values(X)
            base_value = float(explainer.expected_value)

            # Get top features by absolute SHAP value
            sv_flat = shap_vals[0]
            feature_shap = list(zip(feature_names, sv_flat.tolist()))
            feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)
            top_features = feature_shap[:15]

            result[target_name] = {
                "base_value": round(base_value, 4),
                "predicted": round(base_value + float(sv_flat.sum()), 4),
                "features": [
                    {
                        "name": name,
                        "shap_value": round(val, 6),
                        "feature_value": round(combined.get(name, 0.0), 4) if isinstance(combined.get(name, 0.0), (int, float)) else 0.0,
                        "direction": "positive" if val > 0 else "negative",
                    }
                    for name, val in top_features
                ],
            }

        return {"targets": result}

    except ImportError:
        return {"targets": {}, "message": "shap package not installed"}
    except Exception as e:
        return {"targets": {}, "message": f"SHAP computation failed: {str(e)}"}


@app.post("/api/what_if")
async def what_if_simulation(payload: WhatIfPayload):
    """Digital Twin: predict outcomes for user-modified decision variable settings."""
    from orchestration_layer import _surrogate_model
    import numpy as np

    if _surrogate_model is None or not _surrogate_model._is_fitted:
        return {"error": "Surrogate model not available"}

    # Get the current batch state for context features
    bid = payload.batch_id or latest_batch_id
    if not bid:
        return {"error": "No active batch"}

    config = {"configurable": {"thread_id": bid}}
    state_snapshot = compiled_graph.get_state(config)
    if not state_snapshot:
        return {"error": "Batch state not found"}

    state_vals = state_snapshot.values
    telemetry = state_vals.get("current_telemetry", {})

    # Build full feature vector from telemetry + user's what-if settings
    feat_vals = []
    from offline_optimizer import DECISION_VARS
    for fname in _surrogate_model.feature_names:
        if fname in payload.settings:
            feat_vals.append(float(payload.settings[fname]))
        elif fname in DECISION_VARS and fname in state_vals.get("proposed_settings", {}):
            feat_vals.append(float(state_vals["proposed_settings"][fname]))
        elif fname in telemetry:
            feat_vals.append(float(telemetry[fname]))
        else:
            feat_vals.append(0.0)

    X = np.array([feat_vals], dtype=np.float64)
    uq = _surrogate_model.predict_with_uncertainty(X)

    predictions = {}
    for i, target in enumerate(_surrogate_model.target_names):
        predictions[target] = {
            "predicted": float(uq["mean"][0][i]),
            "lower_10": float(uq["lower"][0][i]),
            "upper_90": float(uq["upper"][0][i]),
        }

    # Carbon estimate from predicted power
    power_pred = predictions.get("Power_Consumption_kW", {}).get("predicted", 20.0)
    predictions["carbon_estimate_kg"] = round(power_pred * 0.82 / 60, 4)

    return {"predictions": predictions, "settings_used": payload.settings}


@app.get("/api/briefing")
async def get_ai_briefing(batch_id: str = "LATEST_KNOWN"):
    """Get plain-English AI explanation of the current batch recommendation."""
    global latest_batch_id
    if batch_id == "LATEST_KNOWN":
        if not latest_batch_id:
            return {"briefing": "No active batch. Click 'Run New Batch' on the Dashboard to start."}
        batch_id = latest_batch_id

    config = {"configurable": {"thread_id": batch_id}}
    state_snapshot = compiled_graph.get_state(config)
    if not state_snapshot:
        return {"briefing": "Batch state not found."}

    state_vals = state_snapshot.values
    briefing = state_vals.get("ai_briefing", "AI briefing not yet generated for this batch.")
    return {
        "briefing": briefing,
        "batch_id": batch_id,
        "shap_values": state_vals.get("shap_values", {}),
    }


@app.get("/api/audit_trail")
async def get_audit_trail(last_n: int = 50):
    """Get the hash-chained immutable audit ledger."""
    from orchestration_layer import _audit_ledger
    if _audit_ledger is None:
        return {"records": [], "integrity": {"valid": True, "length": 0}, "iso_summary": {}}
    return {
        "records": _audit_ledger.get_latest(last_n),
        "integrity": _audit_ledger.verify_chain(),
        "iso_summary": _audit_ledger.get_iso_summary(),
    }


@app.get("/api/audit_pdf")
async def download_audit_pdf():
    """Generate and return the PDF audit compliance report."""
    from orchestration_layer import _audit_ledger
    from fastapi.responses import FileResponse
    if _audit_ledger is None:
        raise HTTPException(status_code=500, detail="Audit ledger not initialized")

    output_path = _audit_ledger.export_audit_pdf()
    # Determine content type based on extension
    if output_path.endswith(".pdf"):
        return FileResponse(output_path, media_type="application/pdf", filename="audit_report.pdf")
    else:
        return FileResponse(output_path, media_type="text/plain", filename="audit_report.txt")



@app.get("/api/regulatory_targets")
async def get_regulatory_targets():
    """Get current regulatory compliance targets."""
    from orchestration_layer import _regulatory_targets
    return _regulatory_targets


@app.post("/api/regulatory_targets")
async def set_regulatory_targets(payload: RegulatoryPayload):
    """Update regulatory compliance targets."""
    from orchestration_layer import _regulatory_targets, _carbon_tracker
    
    if payload.max_carbon_per_batch_kg is not None:
        _regulatory_targets["max_carbon_per_batch_kg"] = payload.max_carbon_per_batch_kg
    if payload.max_power_per_batch_kwh is not None:
        _regulatory_targets["max_power_per_batch_kwh"] = payload.max_power_per_batch_kwh
    if payload.min_yield_pct is not None:
        _regulatory_targets["min_yield_pct"] = payload.min_yield_pct
    if payload.min_hardness is not None:
        _regulatory_targets["min_hardness"] = payload.min_hardness
    if payload.max_friability is not None:
        _regulatory_targets["max_friability"] = payload.max_friability
    if payload.emission_factor_name is not None:
        _regulatory_targets["emission_factor_name"] = payload.emission_factor_name
        if _carbon_tracker:
            _carbon_tracker.update_emission_factor(payload.emission_factor_name)

    # Also update carbon tracker regulatory limits
    if _carbon_tracker:
        _carbon_tracker.update_regulatory({
            "max_carbon_per_batch_kg": _regulatory_targets["max_carbon_per_batch_kg"],
            "max_power_per_batch_kwh": _regulatory_targets["max_power_per_batch_kwh"],
        })

    print(f"⚙️ Regulatory targets updated: {_regulatory_targets}")
    return {"status": "success", "targets": _regulatory_targets}


if __name__ == "__main__":
    uvicorn.run("api_gateway:app", host="127.0.0.1", port=8000, reload=True)
