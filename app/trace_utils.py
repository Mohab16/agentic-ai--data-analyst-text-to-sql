import json
from pathlib import Path
from typing import Any, Dict, List

def get_all_traces(trace_dir: Path) -> List[Dict[str, Any]]:
    """List all traces in the directory."""
    traces = []
    for file in trace_dir.glob("*.jsonl"):
        traces.append({
            "task_id": file.stem,
            "filename": file.name,
            "created_at": file.stat().st_mtime,
        })
    # Sort by modification time (newest first)
    return sorted(traces, key=lambda x: x["created_at"], reverse=True)

def get_trace_content(trace_dir: Path, task_id: str) -> List[Dict[str, Any]]:
    """Read and parse the JSONL content of a specific trace."""
    file = trace_dir / f"{task_id}.jsonl"
    if not file.exists():
        return []
    
    steps = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            steps.append(json.loads(line))
    return steps
