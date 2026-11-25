import requests, subprocess, sys, os, time
os.chdir(r'E:\ML-Projects\Agentic AI\AI Data Analyst (Agentic Text-to-SQL)')
proc = subprocess.Popen([sys.executable, '-m', 'app.main', '--serve', '--port', '8000'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(2)
try:
    resp = requests.get('http://127.0.0.1:8000/ui/', timeout=5)
    c = resp.text
    checks = [
        ('toggleStepCard', 'toggleStepCard' in c),
        ('toggleRawJson', 'toggleRawJson' in c),
        ('step-card CSS', '.step-card {' in c),
        ('step-raw-json', '.step-raw-json' in c),
        ('step-details-grid', '.step-details-grid' in c),
        ('tool-chip', '.tool-chip' in c),
    ]
    for name, ok in checks:
        status = "OK" if ok else "MISSING"
        print(f'[{status}] {name}')
finally:
    proc.terminate()