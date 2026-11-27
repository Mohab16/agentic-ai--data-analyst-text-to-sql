"""The planner: turn one business question into a validated investigation plan.

The model's freedom is real but bounded. It decides the metric, the dates, the
comparison baseline, and the checks worth running. It does not decide the shape:
the payload is parsed against a schema in code, and a plan that fails validation
is sent back once with the problems listed. If the second reply fails too, the
step errors — a plausible-looking plan that silently violates the contract is
worse than no plan.

The plan is advisory for the investigator (C-05). It records intent, so a later
reader can tell whether the investigation answered the question that was asked.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from agents.replies import ReplyInvalid, extract_json, repair_instruction
from core.config import Settings
from core.orchestrator import AgentFn
from core.types import AgentResult, Check, InvestigationPlan, Timeframe, to_dict
from core.utils import is_slug as _is_slug
from llm.base import LLMClient, LLMError, LLMRequest, Message

_GRANULARITIES = frozenset({"day", "week", "month"})
_TEXT_LIMIT = 200
_DIMENSION_LIMIT = 40
_MAX_DIMENSIONS = 6

_PLANNER_SYSTEM = """You are the planner of an AI data analyst.

You receive one business question and a database schema digest. You cannot query
the database. Your job is the investigation plan that a separate investigator,
who can run SQL, will carry out.

Reply with exactly one JSON object and nothing else: no prose, no code fences.

{{
  "metric_definition": "one or two sentences: the measure, how it is computed from the tables, and which rows count",
  "metric_type": "gross | net | both",
  "timeframe": {{
    "start": "YYYY-MM-DD",
    "end": "YYYY-MM-DD",
    "comparison": "what the numbers are compared against, or none",
    "granularity": "day | week | month"
  }},
  "answerable": true,
  "checks": [
    {{"id": "c1", "question": "what this check answers", "method": "how to answer it, in words", "dimensions": ["columns to group by"]}}
  ],
  "blocking_gap": null,
  "assumptions": ["anything you had to decide"]
}}

Rules:
- Resolve relative dates against the "as of" date in the schema digest. The
  current month is partial, so "this month" ends on that date, not on the last
  day of the month.
- start and end are real calendar dates, start on or before end.
- metric_definition names the tables and columns a reader needs to reproduce the
  number, and states which rows count (status filter, excluded returns).
- metric_type MUST be one of: "gross" (sum of line_total, no refund subtraction), "net" (sum of line_total minus refunds), or "both" (compute both). 
  If the question asks about "revenue drop", choose "gross" unless refunds are explicitly mentioned.
- Write between 1 and {max_checks} checks, broad first. ids are c1, c2, ...
  A method is a description of the calculation, never SQL: you do not know the
  dialect or the exact column types, and the investigator writes the query.
- If the schema cannot answer the question, set answerable to false, explain the
  missing data in blocking_gap, and leave checks empty. Never invent a table.
- A plan tests hypotheses. It does not conclude causes.
}}"""


class PlanPayloadInvalid(ReplyInvalid):
    """The model's plan payload violates the plan contract."""


def make_planner(client: LLMClient) -> AgentFn:
    """Bind planning to one model client. The returned function is the agent."""

    def planner(context: str, settings: Settings) -> AgentResult:
        system = _PLANNER_SYSTEM.format(max_checks=settings.max_queries)
        messages: tuple[Message, ...] = (Message(role="user", content=context),)
        tokens_in = 0
        tokens_out = 0

        # The planner structurally requires JSON output per its system prompt.
        # Always use json_mode=True; the investigator has a fallback protocol.
        planner_json_mode = True

        for attempt in (1, 2):
            try:
                response = client.complete(
                    LLMRequest(
                        system=system,
                        messages=messages,
                        json_mode=planner_json_mode,
                        max_tokens=settings.llm_max_tokens,
                    )
                )
            except LLMError as exc:
                return AgentResult(
                    status="error",
                    error=f"llm_error: {exc.kind}: {exc.message}",
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                )

            tokens_in += response.tokens_in
            tokens_out += response.tokens_out

            try:
                plan = parse_plan_payload(
                    extract_json(response.text), max_checks=settings.max_queries
                )
            except ReplyInvalid as exc:
                if attempt == 2:
                    return AgentResult(
                        status="error",
                        error="plan_invalid: " + "; ".join(exc.problems),
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                    )
                messages = _repair_turn(context, response.text, exc.problems)
                continue

            return AgentResult(
                status="ok",
                text=render_plan_summary(plan),
                data=to_dict(plan),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )

        return AgentResult(  # pragma: no cover - the loop always returns
            status="error",
            error="plan_invalid: no reply was produced",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    return planner


def parse_plan_payload(payload: Any, *, max_checks: int) -> InvestigationPlan:
    """Build an `InvestigationPlan` or raise with every problem found.

    Problems are collected rather than raised one at a time: the repair turn is
    more effective when the model sees the whole list.
    """
    if not isinstance(payload, dict):
        raise PlanPayloadInvalid(["the reply is not a JSON object"])

    problems: list[str] = []

    metric = _text(payload.get("metric_definition"), "metric_definition", problems)
    metric_type = _metric_type(payload.get("metric_type"), problems)
    timeframe = _timeframe(payload.get("timeframe"), problems)

    raw_answerable = payload.get("answerable")
    answerable = raw_answerable if isinstance(raw_answerable, bool) else None
    if answerable is None:
        problems.append("answerable must be true or false")

    checks = _checks(payload.get("checks"), max_checks, problems)
    gap = payload.get("blocking_gap")
    if gap is not None and not isinstance(gap, str):
        problems.append("blocking_gap must be a string or null")
        gap = None
    gap_text = (gap or "").strip()

    if answerable is True and not checks:
        problems.append("answerable is true, so checks must contain at least one check")
    if answerable is False:
        if not gap_text:
            problems.append(
                "answerable is false, so blocking_gap must say what data is missing"
            )
        if checks:
            problems.append("answerable is false, so checks must be empty")

    assumptions = _text_list(payload.get("assumptions"), "assumptions", problems)

    if problems:
        raise PlanPayloadInvalid(problems)

    return InvestigationPlan(
        metric_definition=metric,
        metric_type=metric_type,
        timeframe=timeframe,
        answerable=bool(answerable),
        checks=tuple(checks),
        blocking_gap=gap_text or None,
        assumptions=tuple(assumptions),
    )


def render_plan_summary(plan: InvestigationPlan) -> str:
    """The plan artifact: what the investigator reads, and what the CLI prints."""
    timeframe = plan.timeframe
    lines = [
        f"metric: {plan.metric_definition}",
        f"metric_type: {plan.metric_type}",
        (
            f"timeframe: {timeframe.start} .. {timeframe.end} "
            f"(comparison: {timeframe.comparison}, granularity: {timeframe.granularity})"
        ),
        f"answerable: {'yes' if plan.answerable else 'no'}",
    ]
    if plan.blocking_gap:
        lines.append(f"blocking gap: {plan.blocking_gap}")

    if not plan.checks:
        lines.append("checks: none")
    else:
        lines.append(f"checks: {len(plan.checks)}")
        width = max(len(check.id) for check in plan.checks)
        for check in plan.checks:
            lines.append(f"  {check.id.ljust(width)}  {check.question}")
            lines.append(f"  {' ' * width}  method: {check.method}")
            if check.dimensions:
                lines.append(
                    f"  {' ' * width}  dimensions: {', '.join(check.dimensions)}"
                )

    if plan.assumptions:
        lines.append("assumptions:")
        lines.extend(f"  - {assumption}" for assumption in plan.assumptions)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Payload parsing
# --------------------------------------------------------------------------


def _timeframe(value: Any, problems: list[str]) -> Timeframe:
    """Parse the timeframe. On failure the problems explain why; the caller raises."""
    if not isinstance(value, dict):
        problems.append("timeframe must be an object with start, end, comparison")
        value = {}

    dates: dict[str, str] = {}
    for key in ("start", "end"):
        raw = value.get(key)
        if not isinstance(raw, str) or not raw.strip():
            problems.append(f"timeframe.{key} must be a date string (YYYY-MM-DD)")
            continue
        try:
            dates[key] = date.fromisoformat(raw.strip()).isoformat()
        except ValueError:
            problems.append(f"timeframe.{key} is not a real date: {raw.strip()!r}")

    if dates.get("start") and dates.get("end") and dates["start"] > dates["end"]:
        problems.append("timeframe.start is after timeframe.end")

    comparison = _text(value.get("comparison"), "timeframe.comparison", problems)
    granularity = value.get("granularity", "day")
    if not isinstance(granularity, str) or granularity.strip().lower() not in _GRANULARITIES:
        problems.append(
            "timeframe.granularity must be one of day, week, month "
            f"(got {granularity!r})"
        )
        granularity = "day"

    return Timeframe(
        start=dates.get("start", ""),
        end=dates.get("end", ""),
        comparison=comparison,
        granularity=str(granularity).strip().lower(),
    )



def _checks(value: Any, max_checks: int, problems: list[str]) -> list[Check]:
    if value is None:
        return []
    if not isinstance(value, list):
        problems.append("checks must be a list")
        return []
    if len(value) > max_checks:
        problems.append(f"checks has {len(value)} entries; at most {max_checks} are allowed")

    checks: list[Check] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        where = f"checks[{index}]"
        if not isinstance(raw, dict):
            problems.append(f"{where} must be an object")
            continue

        identifier = raw.get("id")
        if not isinstance(identifier, str) or not _is_slug(identifier):
            problems.append(f"{where}.id must match [a-z][a-z0-9_]*")
            identifier = f"check_{index}"
        elif identifier in seen:
            problems.append(f"{where}.id is duplicated: {identifier!r}")
        seen.add(identifier)

        question = _text(raw.get("question"), f"{where}.question", problems)
        method = _text(raw.get("method"), f"{where}.method", problems)
        dimensions = _text_list(raw.get("dimensions"), f"{where}.dimensions", problems)
        if len(dimensions) > _MAX_DIMENSIONS:
            problems.append(f"{where}.dimensions has more than {_MAX_DIMENSIONS} entries")
        for dimension in dimensions:
            if len(dimension) > _DIMENSION_LIMIT:
                problems.append(f"{where}.dimensions entry is longer than {_DIMENSION_LIMIT} chars")

        checks.append(
            Check(
                id=identifier,
                question=question,
                method=method,
                dimensions=tuple(dimensions),
            )
        )
    return checks


def _text(value: Any, field: str, problems: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        problems.append(f"{field} must be a non-empty string")
        return ""
    text = value.strip()
    if len(text) > _TEXT_LIMIT:
        problems.append(f"{field} is longer than {_TEXT_LIMIT} characters")
    return text[:_TEXT_LIMIT]


def _text_list(value: Any, field: str, problems: list[str]) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        problems.append(f"{field} must be a list of strings")
        return []
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            problems.append(f"{field}[{index}] must be a non-empty string")
            continue
        items.append(item.strip()[: _TEXT_LIMIT])
    return items


def _metric_type(value: Any, problems: list[str]) -> str:
    """Validate metric_type is one of the allowed values."""
    allowed = {"gross", "net", "both"}
    if value is None or not isinstance(value, str):
        problems.append("metric_type must be one of: gross, net, both")
        return "gross"
    mt = value.strip().lower()
    if mt not in allowed:
        problems.append(f"metric_type must be one of: gross, net, both (got {value!r})")
        return "gross"
    return mt


def _repair_turn(
    context: str, reply: str, problems: tuple[str, ...]
) -> tuple[Message, ...]:
    """Send the rejected reply back with the problems, so the model can fix it."""
    messages = [Message(role="user", content=context)]
    if reply.strip():
        messages.append(Message(role="assistant", content=reply))
    messages.append(
        Message(
            role="user",
            content=repair_instruction(problems, validator="plan validator"),
        )
    )
    return tuple(messages)
