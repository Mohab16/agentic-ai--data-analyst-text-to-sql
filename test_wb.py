import requests, subprocess, sys, os, time
os.chdir(r'E:\ML-Projects\Agentic AI\AI Data Analyst (Agentic Text-to-SQL)')
proc = subprocess.Popen([sys.executable, '-m', 'app.main', '--serve', '--port', '8000'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(2)
try:
    resp = requests.get('http://127.0.0.1:8000/ui/', timeout=5)
    c = resp.text
    checks = [
        ('view-workbench', 'view-workbench' in c),
        ('wb-header', 'wb-header' in c),
        ('wb-main', 'wb-main' in c),
        ('wb-grid', 'wb-grid' in c),
        ('wb-answer-panel', 'wb-answer-panel' in c),
        ('wb-evidence-panel', 'wb-evidence-panel' in c),
        ('wb-progress', 'wb-progress' in c),
        ('wb-progress-step', 'wb-progress-step' in c),
        ('wb-panel-tabs', 'wb-panel-tabs' in c),
        ('tab-plan', 'tab-plan' in c),
        ('tab-sql', 'tab-sql' in c),
        ('tab-plan-checks', 'tab-plan-checks' in c),
        ('switchWorkbenchTab', 'switchWorkbenchTab' in c),
        ('renderEvidence', 'renderEvidence' in c),
        ('renderPlanTab', 'renderPlanTab' in c),
        ('renderSqlTab', 'renderSqlTab' in c),
        ('renderChecksTab', 'renderChecksTab' in c),
    ]
    for name, ok in checks:
        status = "OK" if ok else "MISSING"
        print(f'[{status}] {name}')
finally:
    proc.terminate()