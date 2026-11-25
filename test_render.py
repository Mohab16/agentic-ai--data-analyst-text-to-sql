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
    
    # Get the UI and check the rendered trace list
    ui_resp = requests.get('http://127.0.0.1:8000/ui/', timeout=5)
    content = ui_resp.text
    
    # Find the trace-list section and see what's in it
    start = content.find('id="trace-list"')
    if start != -1:
        end = content.find('</div>', start)
        trace_list_html = content[start:end+6]
        print("Trace list container HTML:")
        print(trace_list_html[:500])
        print("...")
        print()
    
    # Check if trace cards are rendered (they would be after loadTraceList runs)
    # The initial page load calls loadTraceList, so trace cards should be in the HTML
    if 'trace-card' in content:
        print("[OK] trace-card elements found in HTML")
        # Count them
        count = content.count('trace-card')
        print(f"  Number of trace-card occurrences: {count}")
    else:
        print("[MISSING] No trace-card elements in HTML - loadTraceList may not have run or failed")
        
    # Check for filter function
    if 'function filterTraces' in content:
        print("[OK] filterTraces function exists")
    else:
        print("[MISSING] filterTraces function NOT found - search input has no handler!")
        
finally:
    proc.terminate()