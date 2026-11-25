import requests
import time
import subprocess
import sys
import os

os.chdir(r'E:\ML-Projects\Agentic AI\AI Data Analyst (Agentic Text-to-SQL)')
proc = subprocess.Popen([sys.executable, '-m', 'app.main', '--serve', '--port', '8000'], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

try:
    time.sleep(2)
    
    # Test API directly
    print("=== API Tests ===")
    traces_resp = requests.get('http://127.0.0.1:8000/traces', timeout=5)
    traces = traces_resp.json()
    print(f"Traces list: {len(traces)} items")
    
    if traces:
        first_id = traces[0]['task_id']
        print(f"First task: {first_id}")
        
        trace_resp = requests.get(f'http://127.0.0.1:8000/traces/{first_id}', timeout=5)
        steps = trace_resp.json()
        print(f"Trace details: {len(steps)} steps")
        print(f"  Step 1: {steps[0].get('step_id')} - {steps[0].get('agent')}")
        
    # Test UI content
    print("\n=== UI Content Tests ===")
    ui_resp = requests.get('http://127.0.0.1:8000/ui/', timeout=5)
    content = ui_resp.text
    
    # Check trace list renders with click handlers
    if 'onclick="selectTrace(' in content:
        print("[OK] Trace cards have onclick handlers")
    else:
        print("[MISSING] Trace cards missing onclick handlers")
        
    # Check the onclick passes event parameter
    if 'selectTrace(' in content and 'event' in content:
        print("[OK] Event parameter present in onclick")
    else:
        print("[MISSING] Event parameter missing in onclick")
        
finally:
    proc.terminate()