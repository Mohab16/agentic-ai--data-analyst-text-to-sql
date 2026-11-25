import subprocess
import time
import requests
import os
import sys

os.chdir(r"E:\ML-Projects\Agentic AI\AI Data Analyst (Agentic Text-to-SQL)")

# Start server in background
proc = subprocess.Popen([sys.executable, "-m", "app.main", "--serve", "--port", "8002"], 
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

try:
    time.sleep(3)
    # Test traces endpoint
    resp = requests.get("http://127.0.0.1:8002/traces", timeout=5)
    print(f"/traces status: {resp.status_code}")
    data = resp.json()
    print(f"Number of traces: {len(data)}")
    if data:
        first_id = data[0]['task_id']
        print(f"First task_id: {first_id}")
        
        # Test single trace endpoint
        resp2 = requests.get(f"http://127.0.0.1:8002/traces/{first_id}", timeout=5)
        print(f"/traces/{first_id} status: {resp2.status_code}")
        steps = resp2.json()
        print(f"Steps returned: {len(steps)}")
        if steps:
            print(f"First step keys: {list(steps[0].keys())}")
finally:
    proc.terminate()
    proc.wait(timeout=5)