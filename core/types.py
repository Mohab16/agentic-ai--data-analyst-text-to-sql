"""Contracts that cross module boundaries.

Every contract is a frozen dataclass and JSON-serializable through
`to_dict` / `from_dict`. Contracts carry data; behaviour belongs elsewhere.

Rule references: C-01 (dataclasses, JSON-serializable), C-02 (Step.agent is an
explicit key), C-03 (AgentResult.status is explicit), E-01 (typed failures).
"""

from __future__ import annotations

import types
import uuid
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

AgentStatus = Literal["ok", "error"]
TaskStatus = Literal["queued", "running", "done", "error", "cancelled"]
TraceOutcome = Literal["ok", "error", "truncated", "skipped"]
Backend = Literal["mock", "postgres"]


# --------------------------------------------------------------------------
# Failures raised by orchestration / configuration (E-01, E-02)
# --------------------------------------------------------------------------


class ConfigError(Exception):
    """Configuration is missing or invalid. Raised once, at startup (E-06)."""

    def __init__(self, problems: list[str] | tuple[str, ...]) -> None:
        self.problems: tuple[str, ...] = tuple(problems)
        super().__init__("; ".join(self.problems))


class PlanInvalid(Exception):
    """A plan failed structural validation (C-04)."""

    def __init__(self, problems: list[str] | tuple[str, ...]) -> None:
        self.problems: tuple[str, ...] = tuple(problems)
        super().__init__("; ".join(self.problems))


class BudgetExceeded(Exception):
    """A run hit an explicit limit. Raised by the orchestrator (O-03)."""

    def __init__(self, kind: str, limit: float) -> None:
        self.kind = kind
        self.limit = limit
        super().__init__(f"budget_exceeded: {kind} (limit {limit})")


class BackendUnavailable(RuntimeError):
    """The configured database backend cannot serve requests in this build."""


def new_task_id() -> str:
    """A stable id for one investigation run. Generated once, never re-derived."""
    return f"tsk_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# Core contracts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One unit of work. `agent` is a key into the dispatch map, never prose."""

    id: str
    agent: str
    instruction: str
    reads: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    task_id: str
    steps: tuple[Step, ...]
    max_steps: int = 12


@dataclass(frozen=True)
class ToolError:
    """A recoverable tool failure, returned as a value (T-03)."""

    tool: str
    kind: str
    message: str
    retryable: bool = False
    detail: str = ""
    error_id: str | None = None


@dataclass(frozen=True)
class AgentResult:
    """An agent's output. `text` is a deliverable; `error` is a failure (C-03).

    `data` carries the structured payload a machine consumer needs (a plan, an
    investigation report). It must stay JSON-serializable.
    """

    status: AgentStatus
    text: str = ""
    data: dict[str, Any] | None = None
    error: str | None = None
    tool_calls: tuple[str, ...] = ()
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass(frozen=True)
class TaskState:
    """Lifecycle of one run. The orchestrator is the only writer (S-01, S-02)."""

    task_id: str
    status: TaskStatus
    current_step: str | None = None
    error: str | None = None
    steps_completed: int = 0


@dataclass(frozen=True)
class TraceEvent:
    """One event per executed step, whether it succeeds or fails (R-01)."""

    task_id: str
    step_id: str
    agent: str
    model: str
    started_at: str
    duration_ms: int
    outcome: TraceOutcome = "ok"
    prompt_hash: str = ""
    tool_calls: tuple[str, ...] = ()
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None


# --------------------------------------------------------------------------
# Domain payloads carried inside AgentResult.data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Timeframe:
    """The period the question asks about, resolved to concrete dates."""

    start: str
    end: str
    comparison: str
    granularity: str = "day"


@dataclass(frozen=True)
class Check:
    """One intended investigation step. Advisory: the loop may deviate."""

    id: str
    question: str
    method: str
    dimensions: tuple[str, ...] = ()


@dataclass(frozen=True)
class InvestigationPlan:
    """Output of the planner agent."""

    metric_definition: str
    timeframe: Timeframe
    answerable: bool
    metric_type: str = "gross"  # "gross" | "net" | "both"
    checks: tuple[Check, ...] = ()
    blocking_gap: str | None = None
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryRecord:
    """One executed SQL statement, with its outcome. Evidence is addressed by id.

    `preview` is a bounded excerpt of the rows (header + a few rows, capped in
    characters). It exists so a later step can cite a number without re-running
    the query, and so the record stays JSON-serializable and small.
    """

    id: str
    sql: str
    status: AgentStatus
    row_count: int = 0
    duration_ms: int = 0
    truncated: bool = False
    columns: tuple[str, ...] = ()
    error_kind: str | None = None
    error_message: str | None = None
    preview: str = ""


@dataclass(frozen=True)
class Finding:
    """A claim backed by query ids. No claim without evidence."""

    id: str
    claim: str
    evidence_query_ids: tuple[str, ...] = ()
    numbers: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class InvestigationReport:
    """Output of the investigator agent."""

    findings: tuple[Finding, ...] = ()
    queries: tuple[QueryRecord, ...] = ()
    notes: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    stop_reason: str = "model_finished"


@dataclass(frozen=True)
class AnswerFinding:
    """A finding as presented to the user, still carrying its evidence ids."""

    claim: str
    evidence_query_ids: tuple[str, ...] = ()
    numbers: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalAnswer:
    """Output of the reporter agent. Prose only: SQL is rendered by the CLI."""

    headline: str
    answer: str
    findings: tuple[AnswerFinding, ...] = ()
    assumptions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    confidence: str = "medium"


# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------


def to_dict(value: Any) -> Any:
    """Recursively convert contracts into JSON-ready primitives."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_dict(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_dict(item) for key, item in value.items()}
    return value


def from_dict[ContractT](contract: type[ContractT], data: dict[str, Any]) -> ContractT:
    """Rebuild a contract tree from `to_dict` output."""
    return _convert(contract, data)  # type: ignore[return-value]


def _convert(annotation: Any, value: Any) -> Any:
    if value is None:
        return None

    origin = get_origin(annotation)

    if origin is Literal:
        return value

    if origin in (Union, types.UnionType):
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            return _convert(arg, value)
        return value

    if origin in (list, tuple):
        args = get_args(annotation)
        if origin is tuple and args and args[-1] is not Ellipsis:
            return tuple(_convert(arg, item) for arg, item in zip(args, value))
        inner = args[0] if args else Any
        items = [_convert(inner, item) for item in value]
        return tuple(items) if origin is tuple else items

    if origin is dict:
        args = get_args(annotation)
        value_type = args[1] if len(args) == 2 else Any
        return {key: _convert(value_type, item) for key, item in value.items()}

    if is_dataclass(annotation):
        hints = get_type_hints(annotation)
        return annotation(  # type: ignore[operator]
            **{
                f.name: _convert(hints[f.name], value[f.name])
                for f in fields(annotation)
                if f.name in value
            }
        )

    return value
