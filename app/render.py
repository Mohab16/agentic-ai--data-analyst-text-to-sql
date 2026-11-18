"""Terminal rendering. Plain text only: no HTML, no markup dialects.

Rules: P-01 (presentation lives at the transport edge), P-02 (model-derived
text is untrusted; it is printed, never interpreted).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core.config import Settings
from core.types import AgentResult, Plan


def render_planner_output(
    result: AgentResult, *, task_id: str, settings: Settings
) -> str:
    """The plan artifact, plus the run facts a reader needs to trust it.

    The artifact text is exactly what the investigator will receive in phase 4, so
    what you see here is what the next step reads. No second rendering exists.
    """
    trace_file = settings.trace_dir / f"{task_id}.jsonl"
    return "\n".join(
        [
            result.text.strip() or "(the planner returned no text)",
            "",
            f"task:   {task_id}",
            f"model:  {settings.llm_model}",
            f"tokens: {result.tokens_in} in / {result.tokens_out} out",
            f"trace:  {trace_file}",
        ]
    )


def render_answer_output(
    answer: AgentResult | None,
    plan: AgentResult | None,
    investigation: AgentResult | None,
    *,
    task_id: str,
    settings: Settings,
) -> str:
    """The plan, the investigation, the answer, and the run facts. In that order.

    A reader of the answer still needs the evidence behind it: the plan says what
    was checked, the investigation shows what ran, and the answer states what the
    numbers mean. Each section is the artifact text exactly as the next step
    received it; no second rendering exists.
    """
    parts: list[str] = []
    if plan is not None and plan.status == "ok" and plan.text.strip():
        parts.append(f"plan\n----\n{plan.text.strip()}")
    if investigation is not None and investigation.text.strip():
        parts.append(f"investigation\n-------------\n{investigation.text.strip()}")
    if answer is not None and answer.text.strip():
        parts.append(f"answer\n------\n{answer.text.strip()}")

    tokens_in, tokens_out = _tokens(plan, investigation, answer)
    parts.append(_facts(task_id, settings, tokens_in, tokens_out))
    return "\n\n".join(parts)


def _evidence_block(
    investigation: AgentResult | None, settings: Settings
) -> str:
    if investigation is None or not investigation.data:
        return ""
    queries = investigation.data.get("queries")
    if not isinstance(queries, list):
        return ""
    stop_reason = investigation.data.get("stop_reason")
    return render_query_log(
        queries,
        stop_reason=stop_reason if isinstance(stop_reason, str) else None,
        allowed=settings.max_queries,
    )


def _tokens(*results: AgentResult | None) -> tuple[int, int]:
    present = [result for result in results if result is not None]
    return (
        sum(result.tokens_in for result in present),
        sum(result.tokens_out for result in present),
    )


def _facts(
    task_id: str, settings: Settings, tokens_in: int, tokens_out: int
) -> str:
    return "\n".join(
        [
            f"task:   {task_id}",
            f"model:  {settings.llm_model}",
            f"tokens: {tokens_in} in / {tokens_out} out",
            f"trace:  {settings.trace_dir / f'{task_id}.jsonl'}",
        ]
    )


def render_run_output(
    plan: AgentResult | None,
    investigation: AgentResult | None,
    *,
    task_id: str,
    settings: Settings,
) -> str:
    """The plan, the evidence, and the run facts. No section is invented client-side."""
    parts: list[str] = []
    if plan is not None and plan.status == "ok" and plan.text.strip():
        parts.append(f"plan\n----\n{plan.text.strip()}")
    if investigation is not None and investigation.text.strip():
        parts.append(f"investigation\n-------------\n{investigation.text.strip()}")

    tokens_in, tokens_out = _tokens(plan, investigation)
    parts.append(_facts(task_id, settings, tokens_in, tokens_out))
    return "\n\n".join(parts)


def render_query_log(
    queries: Sequence[Mapping[str, Any]],
    *,
    stop_reason: str | None = None,
    allowed: int | None = None,
) -> str:
    """The SQL that ran, from the investigation's structured payload.

    Used by the answer output and by the error path: a run that died after six
    queries should still show the six queries, or the failure hides the evidence it
    paid for.
    """
    if not queries:
        return ""
    if allowed is None:
        header = f"queries attempted: {len(queries)}"
    else:
        header = f"queries: {len(queries)} attempted of {allowed} allowed"
        if stop_reason:
            header += f" (stop reason: {stop_reason})"
    lines = [header]
    for record in queries:
        status = "ok" if record.get("status") == "ok" else str(
            record.get("error_kind") or "error"
        )
        detail = (
            f"{record.get('row_count', 0)} rows"
            if status == "ok"
            else status
        )
        lines.append(f"  {record.get('id', '?')}  {detail}")
        sql = " ".join(str(record.get("sql", "")).split())
        if sql:
            lines.append(f"      {sql[:160]}")
    return "\n".join(lines)


def render_plan(plan: Plan, settings: Settings, question: str) -> str:
    """Human-readable view of a validated plan and the effective limits."""
    rows = [
        (str(index), step.id, step.agent, ", ".join(step.reads) or "-")
        for index, step in enumerate(plan.steps, start=1)
    ]
    headers = ("#", "step", "agent", "reads")
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]

    def line(cells: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(widths[col]) for col, cell in enumerate(cells)).rstrip()

    divider = "  ".join("-" * width for width in widths)
    table = [line(headers), divider, *(line(row) for row in rows)]

    limits = (
        f"steps {settings.max_steps} | tool turns {settings.max_tool_turns} | "
        f"queries {settings.max_queries} | retries {settings.max_query_retries} | "
        f"rows {settings.max_rows} | timeout {settings.statement_timeout_ms}ms | "
        f"context {settings.context_budget} chars | wall {settings.wall_clock_s}s"
    )
    model = settings.llm_model or "(not configured)"
    backend = settings.database_backend
    if backend == "mock":
        backend = (
            f"mock (seed {settings.mock_seed}, {settings.mock_months} months, "
            f"as of {settings.reference_date.isoformat()})"
        )

    return "\n".join(
        [
            f"task:     {plan.task_id}",
            f"question: {question}",
            "",
            f"plan: {len(plan.steps)} steps (max_steps {plan.max_steps})",
            *table,
            "",
            f"backend: {backend}",
            f"model:   {model}",
            f"limits:  {limits}",
            f"trace:   {settings.trace_dir}",
            "",
            "validation: OK (agents are not implemented yet; --dry-run stops here)",
        ]
    )
