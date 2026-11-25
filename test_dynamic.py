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
    
    # Get the UI - this is the INITIAL HTML before JS runs
    ui_resp = requests.get('http://127.0.0.1:8000/ui/', timeout=5)
    content = ui_resp.text
    
    # Check for filter function in JS
    if 'function filterTraces' in content:
        print("[OK] filterTraces function exists")
    else:
        print("[MISSING] filterTraces function NOT found - search input has no handler!")
        
    # The trace cards are rendered DYNAMICALLY by loadTraceList() which runs on DOMContentLoaded
    # The test only sees the initial HTML, not the post-JS DOM
    print("\nNote: trace cards are rendered dynamically via loadTraceList() on page load")
    print("The 2 occurrences found are likely in the JavaScript template strings")
    
finally:
    proc.terminate()