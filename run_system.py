import subprocess
import sys
import os
import time
import webbrowser
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler

def run_backend():
    print("Starting Backend (FastAPI)...")
    project_root = os.getcwd()
    backend_dir = os.path.join(project_root, "backend")
    
    # Using the venv python if available
    python_exe = os.path.join(backend_dir, "venv", "Scripts", "python.exe") if os.name == "nt" else os.path.join(backend_dir, "venv", "bin", "python")
    if not os.path.exists(python_exe):
        python_exe = sys.executable
    
    subprocess.run([python_exe, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"], cwd=backend_dir)

def run_frontend():
    print("Starting Frontend Server (http://localhost:3000)...")
    project_root = os.getcwd()
    frontend_dir = os.path.join(project_root, "frontend")
    
    os.chdir(frontend_dir)
    server_address = ('', 3000)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(2)
        webbrowser.open("http://localhost:3000")
    
    threading.Thread(target=open_browser, daemon=True).start()
    httpd.serve_forever()

if __name__ == "__main__":
    if not os.path.exists("backend") or not os.path.exists("frontend"):
        print("Error: Please run this script from the project root directory.")
        sys.exit(1)

    # Use threading to run both
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()

    # Stay in the root for a bit before moving to frontend
    time.sleep(2)
    
    try:
        run_frontend()
    except KeyboardInterrupt:
        print("\nShutting down system...")
        sys.exit(0)
