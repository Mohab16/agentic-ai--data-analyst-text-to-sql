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
    
    resp = requests.get('http://127.0.0.1:8000/ui/', timeout=5)
    print('UI Status:', resp.status_code)
    
    content = resp.text
    checks = [
        ('view-traces', 'view-traces' in content),
        ('sidebar', 'class="sidebar"' in content),
        ('main-pane', 'class="main-pane"' in content),
        ('trace-list', 'id="trace-list"' in content),
        ('trace-details', 'id="trace-details"' in content),
        ('selectTrace', 'selectTrace' in content),
        ('flex layout', 'display:flex' in content and 'height:100%' in content),
    ]
    
    for name, result in checks:
        status = "OK" if result else "MISSING"
        print(f'  {status}: {name}')
    
finally:
    proc.terminate()