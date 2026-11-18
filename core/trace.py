"""Trace emission: one append-only JSON Lines file per task.

Rules: R-01 (one event per step), R-03 (append-only JSONL, one file per task),
R-04 (logging is additive to the trace, never a replacement), G-05 (redact
secrets).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from core.types import TraceEvent, to_dict

if TYPE_CHECKING:  # pragma: no cover - typing only
    from core.config import Settings

_DSN_PASSWORD = re.compile(r"(://[^:/@\s]+:)[^@\s]+(?=@)")


class TraceSink(Protocol):
    """What the orchestrator needs. A test double satisfies this in three lines."""

    def emit(self, event: TraceEvent) -> None: ...


class Redactor:
    """Removes known secrets and DSN passwords from anything written out."""

    def __init__(self, secrets: Sequence[str] = ()) -> None:
        self._secrets = tuple(s for s in secrets if s and len(s) >= 4)

    def redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "***")
        return _DSN_PASSWORD.sub(r"\1***", text)


class TraceStore:
    """Writes `traces/<task_id>.jsonl`. Append-only; one writer per task."""

    def __init__(self, directory: Path, redactor: Redactor | None = None) -> None:
        self._directory = directory
        self._redactor = redactor or Redactor()

    @classmethod
    def from_settings(cls, settings: Settings) -> TraceStore:
        return cls(settings.trace_dir, Redactor(settings.secrets()))

    def path(self, task_id: str) -> Path:
        return self._directory / f"{task_id}.jsonl"

    def emit(self, event: TraceEvent) -> None:
        target = self.path(event.task_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(to_dict(event), ensure_ascii=False, sort_keys=True)
        with target.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(self._redactor.redact(line) + "\n")

    def read(self, task_id: str) -> list[dict[str, object]]:
        """Read a trace back. Used by debugging, tests, and evaluation fixtures."""
        target = self.path(task_id)
        if not target.exists():
            return []
        with target.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
