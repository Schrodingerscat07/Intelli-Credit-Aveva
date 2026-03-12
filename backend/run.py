import subprocess
import sys
import os
import time
import threading
import io

# Force UTF-8 output to prevent UnicodeEncodeError for emojis on Windows consoles
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

def run_backend():
    print("🚀 Starting FastAPI Backend (Uvicorn)..")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    subprocess.run([sys.executable, "-m", "uvicorn", "api_gateway:app", "--host", "127.0.0.1", "--port", "8000", "--reload"], env=env)

def run_frontend():
    print("🚀 Starting React Frontend (Vite)..")
    # Assuming 'npm' is in the system PATH
    cwd = os.path.join(os.path.dirname(__file__), "..", "frontend")
    vite_bin = os.path.join("node_modules", "vite", "bin", "vite.js")
    
    # Use direct node execution to bypass cmd.exe parsing errors with '&' in folder names
    subprocess.run(["node", vite_bin], cwd=cwd)

if __name__ == "__main__":
    print("=====================================================")
    print("   Intelli-Credit-Aveva — Industrial AI Optimizer    ")
    print("             Phase 4: Full-Stack Start               ")
    print("=====================================================")

    # Start FastAPI
    backend_thread = threading.Thread(target=run_backend)
    backend_thread.daemon = True
    backend_thread.start()

    time.sleep(3) # Give backend a second to boot and load PyTorch

    # Start Vite
    frontend_thread = threading.Thread(target=run_frontend)
    frontend_thread.daemon = True
    frontend_thread.start()

    print("\n✅ System is running!")
    print("👉 Frontend: http://localhost:5173")
    print("👉 Backend:  http://localhost:8000/docs\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        sys.exit(0)
