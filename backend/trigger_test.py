import requests
import time
import json
import random

API_URL = "http://127.0.0.1:8000/api/trigger_batch"

# Let's generate a slightly degraded telemetry signal to force the AI to recommend a fix
mock_telemetry = {
    "Temperature_C": 82.5,  # Too hot
    "Pressure_Bar": 2.1,
    "Humidity_Percent": 40.0,
    "Motor_Speed_RPM": 85.0, # Too fast
    "Compression_Force_kN": 22.0,
    "Flow_Rate_LPM": 150.0,
    "Power_Consumption_kW": 35.0, # High power
    "Vibration_mm_s": 5.2,
    
    # Engineered features needed by the prompt
    "Thermal_Ramp_Rate": 1.5,
    "Power_AUC": 1200.0,
    "Vibration_AUC": 350.0,
    # Injecting the context features to match our 284 cols
}

# The actual model needs 284 feature columns. For the demo, the backend's vector memory 
# dynamically handles whatever we feed it compared to the loaded Golden Signatures.
# We will just pad the dict to 284 keys using standard defaults 
# so the StandardScaler doesn't complain.

print("Generating 284-dimensional mock telemetry profile...")
for i in range(284 - len(mock_telemetry)):
    mock_telemetry[f"Feature_{i}"] = random.normalvariate(0.5, 0.1)

payload = {
    "batch_id": "TEST-BATCH-001",
    "telemetry": mock_telemetry
}

print(f"🚀 Triggering LangGraph optimization batch {payload['batch_id']}...")
response = requests.post(API_URL, json=payload)

if response.status_code == 200:
    print("✅ Successfully triggered batch!")
    print(json.dumps(response.json(), indent=2))
    print("\n👉 Now open the React Dashboard (http://localhost:5173).")
    print("The Dashboard will poll /api/graph_state and pause at the Execution Gate!")
else:
    print(f"❌ Failed to trigger batch: {response.status_code}")
    print(response.text)
