"""Traces are append-only JSONL, one file per task, with secrets redacted (R-03)."""

from __future__ import annotations

import json
from pathlib import Path

from core.config import Settings
from core.trace import Redactor, TraceStore
from core.types import TraceEvent


def event(task_id: str = "tsk_1", step_id: str = "plan", **overrides: object) -> TraceEvent:
    base: dict[str, object] = {
        "task_id": task_id,
        "step_id": step_id,
        "agent": "planner",
        "model": "test-model",
        "started_at": "2026-01-01T00:00:00+00:00",
        "duration_ms": 5,
    }
    base.update(overrides)
    return TraceEvent(**base)  # type: ignore[arg-type]


def test_emit_creates_one_file_per_task(trace_dir: Path) -> None:
    store = TraceStore(trace_dir)
    store.emit(event("tsk_a"))
    store.emit(event("tsk_b"))
    assert store.path("tsk_a").exists()
    assert store.path("tsk_b").exists()
    assert store.path("tsk_a").name == "tsk_a.jsonl"


def test_events_append_as_json_lines(trace_dir: Path) -> None:
    store = TraceStore(trace_dir)
    store.emit(event(step_id="plan"))
    store.emit(event(step_id="investigate"))
    lines = store.path("tsk_1").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["step_id"] == "plan"
    assert json.loads(lines[1])["step_id"] == "investigate"


def test_event_fields_survive_the_roundtrip(trace_dir: Path) -> None:
    store = TraceStore(trace_dir)
    store.emit(event(outcome="error", error="model_call_failed", tokens_in=7, tokens_out=9))
    record = store.read("tsk_1")[0]
    assert record["outcome"] == "error"
    assert record["error"] == "model_call_failed"
    assert record["tokens_in"] == 7
    assert record["tokens_out"] == 9


def test_read_returns_empty_list_for_unknown_task(trace_dir: Path) -> None:
    assert TraceStore(trace_dir).read("tsk_missing") == []


def test_known_secrets_are_redacted(trace_dir: Path) -> None:
    store = TraceStore(trace_dir, Redactor(["sk-test-000000000000"]))
    store.emit(event(error="auth failed for sk-test-000000000000"))
    line = store.path("tsk_1").read_text(encoding="utf-8")
    assert "sk-test-000000000000" not in line
    assert "***" in line


def test_dsn_passwords_are_redacted(trace_dir: Path) -> None:
    store = TraceStore(trace_dir, Redactor())
    store.emit(event(error="connect failed: postgresql://analyst:supersecret@localhost:5432/db"))
    line = store.path("tsk_1").read_text(encoding="utf-8")
    assert "supersecret" not in line
    assert "analyst" in line


def test_short_secrets_are_not_treated_as_secrets(trace_dir: Path) -> None:
    redactor = Redactor(["ab"])
    assert redactor.redact("abracadabra") == "abracadabra"


def test_store_from_settings_uses_configured_directory(
    settings: Settings, trace_dir: Path
) -> None:
    store = TraceStore.from_settings(settings)
    assert store.path("tsk_x") == trace_dir / "tsk_x.jsonl"


def test_store_from_settings_redacts_the_api_key(settings: Settings) -> None:
    store = TraceStore.from_settings(settings)
    store.emit(event(error=f"bad key {settings.llm_api_key}"))
    line = store.path("tsk_1").read_text(encoding="utf-8")
    assert settings.llm_api_key not in line
