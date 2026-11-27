"""The investigator: a bounded tool loop whose output is evidence, not prose.

The model runs SQL through `Toolset` and reads the results. It stops in one of three
ways, and the report records which one: it stops calling tools, it runs out of
turns, or it runs out of queries. The last two still get a final turn without tools,
so a cut-short investigation produces findings from what it has instead of nothing.

Its deliverable is not a narrative. It is a JSON object of claims, each citing the
`query_id` values that support it, and the claims are checked in code: a finding
whose evidence ids were never returned by a query is rejected and sent back once.
That makes "no claim without evidence" a property of the run rather than a promise
in a prompt. Whether the numbers inside a claim match the row values is a question
for the evaluations in phase 8, not for this parser.

The report also carries every `QueryRecord`, including failures and refusals, so the
next step can render the SQL that ran and say where the investigation stopped.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from typing import Any

from agents.replies import ReplyInvalid, extract_json, repair_instruction
from core.config import Settings
from core.orchestrator import AgentFn
from core.types import (
    AgentResult,
    Finding,
    InvestigationReport,
    QueryRecord,
    to_dict,
)
from core.utils import is_slug as _is_slug
from llm.base import LLMClient, LLMError, LLMRequest, Message
from tools.registry import Toolset

_FINDINGS_MAX = 10
_CLAIM_LIMIT = 400
_TEXT_LIMIT = 200
_MAX_NUMBERS = 8
_SQL_SHOWN = 160
# Keep at most this many recent messages in the conversation window. Older
# tool results are dropped to keep the prompt small on long investigations.
_MAX_HISTORY = 12
_MAX_CONVERSATION_TOKENS = 15000  # Maximum tokens in conversation before pruning

_SYSTEM = """You are the investigator of an AI data analyst.

You receive one business question, the plan for answering it, and a database schema
digest. You answer by running SQL through execute_sql, one statement at a time, and
reading the results. If you did not see a number in a result, you do not report it.

Evidence rules:
- A change is not a cause. Say what moved, list candidate explanations only as
  hypotheses, and say which query would test each one.
- The current month is partial. Compare like with like and never treat a partial
  period as a full one.
- A zero-row result is a finding: report it as such.
- Results are capped at {max_rows} rows and cancelled after {timeout_ms} ms. When a
  result is truncated, aggregate in SQL instead of trusting the rows you saw.
- When a query fails, read the message and use get_query_error_details if that is
  not enough. Fix the statement; do not resend it unchanged.
- Prefer the schema digest, and use inspect_schema only for detail you need.
- Division: in SQLite, integer division truncates. Always write ratios/rates as
  `CAST(numerator AS REAL) / CAST(denominator AS REAL)` or `1.0 * numerator / denominator`
  so the result is a float. Never write `a / b` when both are integer columns/expressions.
- Rate calculations (CRITICAL - this is the #1 source of wrong answers):
  To compute ANY rate (return rate, refund rate, cancellation rate, etc.):
  
  1. ALWAYS start FROM the denominator table (the "all items" table)
     - Return rate: FROM order_items (all sold items)
     - Refund rate: FROM orders (all placed orders)
  
  2. ALWAYS LEFT JOIN the numerator table (the "subset" table)
     - Return rate: LEFT JOIN returns ON returns.order_item_id = order_items.id
     - Refund rate: LEFT JOIN refunds ON refunds.order_id = orders.id
  
  3. ALWAYS use COALESCE on the numerator aggregate: COALESCE(SUM(r.quantity), 0)
  
  4. ALWAYS GROUP BY the dimension from the DENOMINATOR table
     - Return rate: GROUP BY p.category (from products via order_items)
  
  WRONG PATTERN (produces 100% or inflated rates - ONLY sees returned items):
    SELECT p.category, SUM(r.quantity)/SUM(oi.quantity) AS rate
    FROM returns r
    JOIN order_items oi ON r.order_item_id = oi.id
    JOIN products p ON oi.product_id = p.id
    GROUP BY p.category
    -- Result: Sports 100%, Electronics 84%, ... (WRONG!)
  
  RIGHT PATTERN (correct rate - sees ALL items including non-returned):
    SELECT p.category,
           CAST(COALESCE(SUM(r.quantity), 0) AS REAL) / CAST(SUM(oi.quantity) AS REAL) AS rate
    FROM order_items oi
    JOIN products p ON oi.product_id = p.id
    LEFT JOIN returns r ON r.order_item_id = oi.id
    GROUP BY p.category
    -- Result: Fashion 6.4%, Sports 5.4%, ... (CORRECT!)
  
  NEVER start FROM returns, refunds, cancellations, etc. for rate calculations.
  ALWAYS start FROM the "all items" table and LEFT JOIN the subset.

Budget: at most {max_queries} queries and {max_tool_turns} turns. Few broad queries
beat many narrow ones.

When you have enough evidence, stop calling tools and reply with exactly one JSON
object and nothing else:

{{
  "findings": [
    {{"id": "f1",
     "claim": "Revenue fell 11.8% in October 2026 versus September (81,422.47 to 71,811.05).",
     "evidence_query_ids": ["q_001", "q_002"],
     "numbers": {{"october_revenue": 71811.05, "september_revenue": 81422.47, "change_pct": -11.8}}}}
  ],
  "open_questions": ["what the evidence cannot answer, and what data would"],
  "notes": ["what a reader needs in order to read these numbers correctly"]
}}

Rules for that object: evidence_query_ids are the query_id values you saw in the
results, and a claim without evidence is rejected; numbers are values you read, not
estimates; a claim is a sentence about the data, never a cause you did not test.

Every number in a finding must appear EXACTLY as a value in the rows of a cited
query. Do not compute, subtract, round, or derive any new number. If you need a
comparison, describe the relationship in prose using the raw numbers from the
queries. A finding whose numbers were not returned by a query is rejected and sent
back once.

If the plan says the question cannot be answered with this schema, run no queries:
return "findings": [] and put the missing data in open_questions."""

_FINAL_ASK = """\
The investigation budget is exhausted, so no further queries are accepted.
Return your findings now as the single JSON object described in your instructions,
citing only the query_id values you actually saw. If the evidence is not enough to
answer the question, say so in open_questions instead of inventing numbers."""


class FindingsInvalid(ReplyInvalid):
    """The findings payload violates the report contract."""


def make_investigator(client: LLMClient, toolset: Toolset) -> AgentFn:
    """Bind the loop to one model client and one per-run toolset."""

    def investigator(context: str, settings: Settings) -> AgentResult:
        system = _SYSTEM.format(
            max_queries=settings.max_queries,
            max_tool_turns=settings.max_tool_turns,
            max_rows=settings.max_rows,
            timeout_ms=settings.statement_timeout_ms,
        )
        messages: list[Message] = [Message(role="user", content=context)]
        tool_names: list[str] = []
        tokens_in = 0
        tokens_out = 0

        for _turn in range(settings.max_tool_turns):
            try:
                response = client.complete(
                    LLMRequest(
                        system=system,
                        messages=tuple(messages),
                        tools=toolset.specs,
                        max_tokens=settings.llm_max_tokens,
                    )
                )
            except LLMError as exc:
                return _failure(
                    f"llm_error: {exc.kind}: {exc.message}",
                    toolset,
                    tokens_in,
                    tokens_out,
                    stop_reason="llm_error",
                    tool_names=tool_names,
                )

            tokens_in += response.tokens_in
            tokens_out += response.tokens_out

            if not response.tool_calls:
                return _finalize(
                    response.text,
                    messages,
                    client=client,
                    system=system,
                    settings=settings,
                    toolset=toolset,
                    tool_names=tool_names,
                    tokens=(tokens_in, tokens_out),
                    stop_reason="model_finished",
                )

            messages.append(
                Message(
                    role="assistant",
                    content=response.text,
                    tool_calls=response.tool_calls,
                )
            )
            for call in response.tool_calls:
                tool_names.append(call.name)
                outcome = toolset.call(call)
                messages.append(
                    Message(
                        role="tool",
                        content=outcome.text,
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )

            # Trim the conversation window so a long investigation does not
            # blow past the token budget. The system prompt and the most
            # recent exchanges are always kept.
            if len(messages) > _MAX_HISTORY:
                messages = messages[-_MAX_HISTORY:]
            
            # Prune old tool results to keep conversation within token budget
            if sum(len(m.content) for m in messages) > _MAX_CONVERSATION_TOKENS:
                # Keep only the most recent 8 messages
                messages = messages[-8:]

            if toolset.budget_exhausted:
                return _ask_for_findings(
                    messages,
                    client=client,
                    system=system,
                    settings=settings,
                    toolset=toolset,
                    tool_names=tool_names,
                    tokens=(tokens_in, tokens_out),
                    stop_reason="query_budget",
                )

        return _ask_for_findings(
            messages,
            client=client,
            system=system,
            settings=settings,
            toolset=toolset,
            tool_names=tool_names,
            tokens=(tokens_in, tokens_out),
            stop_reason="turn_budget",
        )

    return investigator


def _ask_for_findings(
    messages: list[Message],
    *,
    client: LLMClient,
    system: str,
    settings: Settings,
    toolset: Toolset,
    tool_names: list[str],
    tokens: tuple[int, int],
    stop_reason: str,
) -> AgentResult:
    """One last turn, without tools, so a cut-short run still reports."""
    tokens_in, tokens_out = tokens
    messages.append(Message(role="user", content=_FINAL_ASK))
    try:
        response = client.complete(
            LLMRequest(
                system=system,
                messages=tuple(messages),
                json_mode=True,
                max_tokens=settings.llm_max_tokens,
            )
        )
    except LLMError as exc:
        return _failure(
            f"llm_error: {exc.kind}: {exc.message}",
            toolset,
            tokens_in,
            tokens_out,
            stop_reason="llm_error",
            tool_names=tool_names,
        )

    return _finalize(
        response.text,
        messages,
        client=client,
        system=system,
        settings=settings,
        toolset=toolset,
        tool_names=tool_names,
        tokens=(tokens_in + response.tokens_in, tokens_out + response.tokens_out),
        stop_reason=stop_reason,
    )


def _finalize(
    reply: str,
    messages: list[Message],
    *,
    client: LLMClient,
    system: str,
    settings: Settings,
    toolset: Toolset,
    tool_names: list[str],
    tokens: tuple[int, int],
    stop_reason: str,
) -> AgentResult:
    """Parse the findings; on a contract violation, allow exactly one repair."""
    tokens_in, tokens_out = tokens

    try:
        report = _build_report(reply, toolset, settings, stop_reason)
    except ReplyInvalid as exc:
        repair = list(messages)
        if reply.strip():
            repair.append(Message(role="assistant", content=reply))
        valid_ids = ", ".join(
            record.id for record in toolset.records if record.status == "ok"
        ) or "none"
        repair.append(
            Message(
                role="user",
                content=repair_instruction(
                    exc.problems,
                    validator="findings validator",
                    valid_ids=valid_ids,
                ),
            )
        )
        try:
            response = client.complete(
                LLMRequest(
                    system=system,
                    messages=tuple(repair),
                    json_mode=True,
                    max_tokens=settings.llm_max_tokens,
                )
            )
        except LLMError as error:
            return _failure(
                f"llm_error: {error.kind}: {error.message}",
                toolset,
                tokens_in,
                tokens_out,
                stop_reason="llm_error",
                tool_names=tool_names,
            )

        tokens_in += response.tokens_in
        tokens_out += response.tokens_out
        try:
            report = _build_report(response.text, toolset, settings, stop_reason)
        except ReplyInvalid as second:
            return _failure(
                "report_invalid: " + "; ".join(second.problems),
                toolset,
                tokens_in,
                tokens_out,
                stop_reason="report_invalid",
                tool_names=tool_names,
            )

    return AgentResult(
        status="ok",
        text=render_investigation(report, queries_allowed=settings.max_queries),
        data=to_dict(report),
        tool_calls=tuple(tool_names),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )


def _build_report(
    reply: str, toolset: Toolset, settings: Settings, stop_reason: str
) -> InvestigationReport:
    evidence = {record.id for record in toolset.records if record.status == "ok"}
    findings, open_questions, notes = parse_findings_payload(
        extract_json(reply), evidence=evidence
    )
    _validate_numbers(findings, toolset)
    return InvestigationReport(
        findings=findings,
        queries=tuple(toolset.records),
        notes=notes + _harness_notes(toolset, settings, stop_reason),
        open_questions=open_questions,
        stop_reason=stop_reason,
    )


def _validate_numbers(
    findings: tuple[Finding, ...], toolset: Toolset
) -> None:
    """Every number in a finding must appear in the rows of a cited query.

    The evidence check in `parse_findings_payload` only confirms that a finding
    cites a query that ran. This checks the stronger claim: the number was read
    from that query's result, not invented. On failure it raises `FindingsInvalid`
    so the investigator gets exactly one repair turn.
    
    Zero-row results are valid findings; if a cited query returned zero rows,
    the finding must have no numbers (which is valid for reporting absence).
    """
    by_query = toolset.numbers_by_query()
    for finding in findings:
        if not finding.numbers:
            continue
        allowed = [
            number
            for query_id in finding.evidence_query_ids
            for number in by_query.get(query_id, ())
        ]
        # If allowed is empty, the cited queries returned zero rows.
        # In that case, the finding should have no numbers; if it does, that's an error.
        if not allowed:
            raise FindingsInvalid(
                [
                    f"{finding.id} cites queries that returned zero rows, "
                    f"so it cannot have numbers"
                ]
            )
        for key, value in finding.numbers.items():
            if not any(_same_number_exact(value, number) for number in allowed):
                available = ", ".join(
                    f"{number:.10g}" for number in sorted(set(allowed))
                ) or "none"
                raise FindingsInvalid(
                    [
                        f"{finding.id}.numbers[{key!r}]={value:.10g} is not in the "
                        f"rows of any cited query; available: {available}"
                    ]
                )


def _same_number_exact(left: float, right: float) -> bool:
    """Compare numbers exactly, preserving full precision from query results.
    
    Unlike _same_number (which uses 9g for tolerance), this requires exact match
    to ensure findings copy numbers exactly from query rows without rounding.
    """
    return f"{left:.10g}" == f"{right:.10g}"


def _harness_notes(
    toolset: Toolset, settings: Settings, stop_reason: str
) -> tuple[str, ...]:
    """Facts about the run the model cannot know, and the next step must."""
    notes: list[str] = []
    if stop_reason == "query_budget":
        notes.append(
            "the investigation stopped at the query budget "
            f"({toolset.queries_used} of {settings.max_queries} queries used)"
        )
    elif stop_reason == "turn_budget":
        notes.append(
            f"the investigation stopped at the turn budget "
            f"({settings.max_tool_turns} turns)"
        )
    if toolset.refused:
        notes.append(
            f"{toolset.refused} query attempt(s) were refused by the harness and "
            "never ran"
        )
    if toolset.failures:
        notes.append(
            f"{toolset.failures} query attempt(s) failed; see the query log for why"
        )
    return tuple(notes)


def _failure(
    message: str,
    toolset: Toolset,
    tokens_in: int,
    tokens_out: int,
    *,
    stop_reason: str,
    tool_names: list[str],
) -> AgentResult:
    """A failed run still returns what it learned, so the SQL trail is not lost."""
    return AgentResult(
        status="error",
        error=message,
        data=to_dict(
            InvestigationReport(
                queries=tuple(toolset.records),
                stop_reason=stop_reason,
            )
        ),
        tool_calls=tuple(tool_names),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )


# --------------------------------------------------------------------------
# Findings payload
# --------------------------------------------------------------------------


def parse_findings_payload(
    payload: Any, *, evidence: Collection[str]
) -> tuple[tuple[Finding, ...], tuple[str, ...], tuple[str, ...]]:
    """Return `(findings, open_questions, notes)` or raise with every problem."""
    if not isinstance(payload, dict):
        raise FindingsInvalid(["the reply is not a JSON object"])

    problems: list[str] = []
    findings = _findings(payload.get("findings"), evidence, problems)
    open_questions = _strings(
        payload.get("open_questions"), "open_questions", problems
    )
    notes = _strings(payload.get("notes"), "notes", problems)

    if not findings and not open_questions and not notes:
        problems.append(
            "the reply says nothing: findings, open_questions, and notes are all empty"
        )

    if problems:
        raise FindingsInvalid(problems)
    return tuple(findings), tuple(open_questions), tuple(notes)


def _findings(
    value: Any, evidence: Collection[str], problems: list[str]
) -> list[Finding]:
    if value is None:
        return []
    if not isinstance(value, list):
        problems.append("findings must be a list")
        return []
    if len(value) > _FINDINGS_MAX:
        problems.append(f"findings has more than {_FINDINGS_MAX} entries")

    findings: list[Finding] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        where = f"findings[{index}]"
        if not isinstance(raw, dict):
            problems.append(f"{where} must be an object")
            continue

        identifier = raw.get("id")
        if not isinstance(identifier, str) or not _is_slug(identifier):
            problems.append(f"{where}.id must match [a-z][a-z0-9_]*")
            identifier = f"f{index + 1}"
        elif identifier in seen:
            problems.append(f"{where}.id is duplicated: {identifier!r}")
        seen.add(identifier)

        claim = raw.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            problems.append(f"{where}.claim must be a non-empty string")
            claim = ""
        elif len(claim.strip()) > _CLAIM_LIMIT:
            problems.append(f"{where}.claim is longer than {_CLAIM_LIMIT} characters")
            claim = claim.strip()[:_CLAIM_LIMIT]

        citations = raw.get("evidence_query_ids")
        cited: list[str] = []
        if not isinstance(citations, list) or not citations:
            problems.append(
                f"{where}.evidence_query_ids must list at least one query_id "
                "that returned rows"
            )
        else:
            for citation in citations:
                if not isinstance(citation, str) or not citation.strip():
                    problems.append(f"{where}.evidence_query_ids entries must be strings")
                    continue
                if citation.strip() not in evidence:
                    problems.append(
                        f"{where} cites {citation.strip()!r}, which is not a query "
                        "that returned results"
                    )
                    continue
                cited.append(citation.strip())

        numbers = _numbers(raw.get("numbers"), where, problems)

        findings.append(
            Finding(
                id=identifier,
                claim=claim.strip(),
                evidence_query_ids=tuple(cited),
                numbers=numbers,
            )
        )
    return findings


def _numbers(value: Any, where: str, problems: list[str]) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        problems.append(f"{where}.numbers must be an object of name -> number")
        return {}
    if len(value) > _MAX_NUMBERS:
        problems.append(f"{where}.numbers has more than {_MAX_NUMBERS} entries")

    numbers: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not _is_slug(key):
            problems.append(f"{where}.numbers keys must match [a-z][a-z0-9_]*")
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            problems.append(f"{where}.numbers[{key!r}] must be a number")
            continue
        if not math.isfinite(float(raw)):
            problems.append(f"{where}.numbers[{key!r}] must be finite")
            continue
        numbers[key] = float(raw)
    return numbers


def _strings(value: Any, field: str, problems: list[str]) -> list[str]:
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
        items.append(item.strip()[:_TEXT_LIMIT])
    return items


def _same_number(left: float, right: float) -> bool:
    """Compare numbers with tolerance for float noise but preserving sign.
    
    Nine significant digits absorb JSON round-tripping while still refusing a
    rounded or reshaped figure. Unlike the reporter's comparison, this version
    preserves sign to catch attempts to negate numbers.
    """
    return f"{left:.9g}" == f"{right:.9g}"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_investigation(report: InvestigationReport, *, queries_allowed: int) -> str:
    """The investigation artifact: what the reporter reads and the CLI prints.

    The text half is kept compact so a full investigation fits within the context
    budget; the structured payload carries everything in machine-readable form.
    """
    lines: list[str] = []
    if report.findings:
        lines.append(f"findings: {len(report.findings)}")
        for finding in report.findings:
            cited = ", ".join(finding.evidence_query_ids) or "no evidence"
            lines.append(f"  {finding.id}  [{cited}]  {finding.claim}")
            if finding.numbers:
                rendered = ", ".join(
                    f"{name}={_number(value)}" for name, value in finding.numbers.items()
                )
                lines.append(f"      numbers: {rendered}")
    else:
        lines.append("findings: none")

    if report.open_questions:
        lines.append("open questions:")
        lines.extend(f"  - {question}" for question in report.open_questions)
    if report.notes:
        lines.append("notes:")
        lines.extend(f"  - {note}" for note in report.notes)

    lines.append(f"queries: {len(report.queries)} attempted of {queries_allowed} allowed")
    for record in report.queries:
        lines.append(f"  {record.id}  {_query_outcome(record)}")
        lines.append(f"      {_one_line(record.sql, _SQL_SHOWN)}")
    lines.append(f"stop reason: {report.stop_reason}")
    return "\n".join(lines)


def _query_outcome(record: QueryRecord) -> str:
    if record.status == "ok":
        truncated = ", truncated" if record.truncated else ""
        return f"ok, {record.row_count} rows{truncated}"
    reason = record.error_kind or "error"
    message = record.error_message or ""
    return f"{reason}: {_one_line(message, 120)}" if message else reason


def _number(value: float) -> str:
    return f"{value:.10g}"


def _one_line(text: str, limit: int) -> str:
    collapsed = " ".join((text or "").split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"
