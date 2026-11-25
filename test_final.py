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
    
    print("=== COMPREHENSIVE TRACES PAGE TEST ===\n")
    
    # 1. API Tests
    print("1. API ENDPOINTS")
    traces_resp = requests.get('http://127.0.0.1:8000/traces', timeout=5)
    traces = traces_resp.json()
    print(f"   GET /traces: {len(traces)} traces - OK")
    
    if traces:
        first_id = traces[0]['task_id']
        trace_resp = requests.get(f'http://127.0.0.1:8000/traces/{first_id}', timeout=5)
        steps = trace_resp.json()
        print(f"   GET /traces/{first_id}: {len(steps)} steps - OK")
    
    # 2. UI Content Tests
    print("\n2. UI CONTENT")
    ui_resp = requests.get('http://127.0.0.1:8000/ui/', timeout=5)
    content = ui_resp.text
    
    checks = [
        ('view-traces section', 'view-traces' in content),
        ('sidebar element', 'class="sidebar"' in content),
        ('main-pane element', 'class="main-pane"' in content),
        ('trace-list container', 'id="trace-list"' in content),
        ('trace-details container', 'id="trace-details"' in content),
        ('search input', 'id="trace-search"' in content),
        ('search has oninput', 'oninput="filterTraces()"' in content),
        ('selectTrace function', 'selectTrace' in content),
        ('onclick passes event', "onclick=\"selectTrace('" in content and "event\"" in content),
        ('filterTraces function', 'function filterTraces' in content),
        ('renderTraceDetails function', 'renderTraceDetails' in content),
        ('loadTraceList function', 'loadTraceList' in content),
        ('flex layout', 'display:flex' in content and 'height:100%' in content),
    ]
    
    all_ok = True
    for name, result in checks:
        status = "OK" if result else "FAIL"
        if not result: all_ok = False
        print(f"   [{status}] {name}")
    
    # 3. Flex Layout Structure
    print("\n3. LAYOUT STRUCTURE")
    if 'view-traces' in content and 'display:flex' in content:
        print("   [OK] view-traces uses flex layout")
    else:
        print("   [FAIL] Missing flex layout")
        all_ok = False
    
    if 'flex:1' in content or 'flex-grow' in content:
        print("   [OK] main-pane has flex grow")
    else:
        print("   [WARN] main-pane may not grow properly")
    
    if 'flex-shrink:0' in content:
        print("   [OK] sidebar has flex-shrink:0")
    else:
        print("   [WARN] sidebar may shrink")
    
    print(f"\n=== RESULT: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'} ===")
    
finally:
    proc.terminate()