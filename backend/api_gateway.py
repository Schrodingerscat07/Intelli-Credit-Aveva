import os
import random
import uuid
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional

# Phase 2 imports
from orchestration_layer import (
    initialize_system,
    compile_graph,
    ManufacturingState,
    DECISION_BOUNDS
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
        "simulated_outcome": state_vals.get("simulated_outcome", {}),
        "quality_delta": state_vals.get("quality_delta", 0.0),
        "qdrant_updated": state_vals.get("qdrant_updated", False),
        "bounds": DECISION_BOUNDS
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
    """Generate a new batch with random telemetry and trigger the optimization graph.
    
    This allows the frontend dashboard to restart the full workflow cycle 
    without needing to run trigger_test.py from the command line.
    """
    global latest_batch_id
    # Generate a unique batch ID
    batch_id = f"BATCH-{uuid.uuid4().hex[:6].upper()}"
    latest_batch_id = batch_id
    
    # Build realistic 284-dimensional mock telemetry
    mock_telemetry = {
        "Temperature_C": round(random.uniform(60, 95), 1),
        "Pressure_Bar": round(random.uniform(1.0, 3.0), 2),
        "Humidity_Percent": round(random.uniform(25, 60), 1),
        "Motor_Speed_RPM": round(random.uniform(40, 100), 1),
        "Compression_Force_kN": round(random.uniform(8, 25), 1),
        "Flow_Rate_LPM": round(random.uniform(80, 200), 1),
        "Power_Consumption_kW": round(random.uniform(15, 50), 1),
        "Vibration_mm_s": round(random.uniform(2, 8), 2),
        "Thermal_Ramp_Rate": round(random.uniform(0.5, 3.0), 2),
        "Power_AUC": round(random.uniform(500, 2000), 1),
        "Vibration_AUC": round(random.uniform(100, 500), 1),
    }
    
    # Pad to 284 features to match the Golden Signature context dimension
    for i in range(284 - len(mock_telemetry)):
        mock_telemetry[f"Feature_{i}"] = round(random.gauss(0.5, 0.1), 4)
    
    background_tasks.add_task(run_graph_background, batch_id, mock_telemetry)
    print(f"🆕 New batch triggered from dashboard: {batch_id}")
    return {"status": "started", "batch_id": batch_id}

@app.post("/api/update_priorities")
async def update_priorities(payload: PriorityPayload):
    """Endpoint for updating optimization priorities from the frontend slider."""
    print(f"🎚️ Priority updated: {payload.priority_type} = {payload.priority_value}")
    # In a full implementation, this might update a global setting or feed into the next graph run
    return {"status": "success", "priority_value": payload.priority_value}


if __name__ == "__main__":
    uvicorn.run("api_gateway:app", host="127.0.0.1", port=8000, reload=True)
