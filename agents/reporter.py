"""The reporter: evidence in, answer out, and the citations are checked.

The reporter is the only agent whose output a business reader sees, and the only
one that can turn a checked investigation into an unchecked claim. Two rules keep
that from happening, both enforced in `parse_final_answer` rather than asked for in
the prompt:

- every `evidence_query_ids` entry must name a query that returned rows, and
- every number the answer asserts must be a number that appears in the evidence
  that those same queries support.

So the answer cannot introduce a figure the investigation did not establish. It can
still be wrong about what the figure *means* - that is what the limitations and the
golden evaluations are for - but it cannot be wrong about where the figure came
from.

The evidence arrives as the context text, the same text the model reads; the
structured half of it is read back with `core.context.read_artifact`, so the check
uses exactly what the model saw and never a private copy.
"""

from __future__ import annotations

import math
from typing import Any

from agents.replies import ReplyInvalid, extract_json, repair_instruction
from core.config import Settings
from core.context import read_artifact
from core.orchestrator import AgentFn
from core.types import AgentResult, AnswerFinding, FinalAnswer, to_dict
from core.utils import is_slug as _is_slug
from llm.base import LLMClient, LLMError, LLMRequest, Message

_HEADLINE_LIMIT = 100
_ANSWER_LIMIT = 2000
_FINDINGS_MAX = 10
_CLAIM_LIMIT = 600
_TEXT_LIMIT = 300
_MAX_NUMBERS = 8
_CONFIDENCE = ("low", "medium", "high")

_SYSTEM = """You are the reporter of an AI data analyst.

You receive the question, the plan, and an investigation report: findings that each
cite the query ids that produced them, the queries that ran with their outcomes, and
the gaps the investigator recorded. You write the answer a business reader needs.

What the answer contains:
- The headline: one sentence, the finding and its number, no hedging.
- Two to four sentences of what the evidence shows, in order: the change, where it
  is concentrated, what it is not. Then the caveats: what the data cannot separate.
- Findings: each claim with the query ids that support it and the numbers it uses.
- Assumptions and limitations: what a reader must know to read the numbers, and what
  this data cannot settle. Every answer has at least one limitation.

Rules that are checked in code, not trusted:
- evidence_query_ids must be query ids from the investigation that returned rows.
- Every number in a finding must appear in the investigation finding that cites the
  same query. Do not compute new numbers, round into a new number, or carry a number
  over from a query that is not cited. If you need a cut the evidence does not have,
  say what is missing in the answer or in limitations instead of estimating it.
- For a finding's `numbers`, copy only values from the investigator finding(s) named
  by that finding's `evidence_query_ids`. Query previews, SQL output, and numbers
  from other findings are not allowed substitutes. If no approved value supports a
  number, omit that number key and describe the limitation instead.
- confidence is "high" only when the investigation's stop reason is exactly
  "model_finished" and the plan said the question is answerable. If the stop reason
  is anything else (e.g., "query_budget", "max_turns"), use "medium" or "low".
  It must be "low" when the plan said the schema cannot answer the question, or
  when the investigation produced no findings. Otherwise "medium".
- Every limitation must be specific to this answer: a partial period, a metric
  definition, a missing table, or correlation mistaken for cause. "Results may vary"
  is not a limitation.

Never present a cause as established. "North sold no Electronics in October" is
evidence. "A stock-out caused it" is a hypothesis, and must be written as one.

Do not paste SQL: the caller prints the queries that ran. Do not address the
engineer; address the person who asked the question.

Reply with exactly one JSON object and nothing else:

{
  "headline": "Revenue fell 8.0% in October 2026, driven by Electronics in the North",
  "answer": "Two to four sentences of prose, using the numbers from the findings.",
  "findings": [
    {"claim": "...", "evidence_query_ids": ["q_001"], "numbers": {"october_revenue": 71811.05}}
  ],
  "assumptions": ["..."],
  "limitations": ["..."]
}
"""


class AnswerInvalid(ReplyInvalid):
    """The answer payload violates the report contract."""


def make_reporter(client: LLMClient) -> AgentFn:
    """One model call, one repair. The evidence is read from the context."""

    def reporter(context: str, settings: Settings) -> AgentResult:
        evidence = read_artifact(context, "investigate")
        if evidence is None:
            return AgentResult(
                status="error",
                error=(
                    "evidence_unavailable: the investigation payload is not in the "
                    "context; it was absent or cut by CONTEXT_BUDGET, so citations "
                    "cannot be checked"
                ),
            )
        plan = read_artifact(context, "plan") or {}

        messages: list[Message] = [Message(role="user", content=context)]
        try:
            response = client.complete(
                LLMRequest(
                    system=_SYSTEM,
                    messages=tuple(messages),
                    json_mode=True,
                    max_tokens=settings.llm_max_tokens,
                )
            )
        except LLMError as exc:
            return AgentResult(
                status="error", error=f"llm_error: {exc.kind}: {exc.message}"
            )

        tokens_in = response.tokens_in
        tokens_out = response.tokens_out

        try:
            answer = parse_final_answer(
                extract_json(response.text), evidence=evidence, plan=plan
            )
        except ReplyInvalid as exc:
            # Build available numbers info from evidence for the repair instruction
            citable = _citable_queries(evidence)
            by_query = _numbers_by_query(evidence)
            available_parts = []
            for query_id in citable:
                if query_id in by_query and by_query[query_id]:
                    nums = ", ".join(f"{n:.10g}" for n in sorted(set(by_query[query_id])))
                    available_parts.append(f"  {query_id}: {nums}")
            available_str = "\n".join(available_parts) if available_parts else "  (none)"
            
            repair_instruction_text = repair_instruction(
                exc.problems,
                validator="answer validator",
                allowed_values="confidence: 'low', 'medium', or 'high'",
            )
            repair = [
                Message(role="user", content=context),
                Message(role="assistant", content=response.text),
                Message(
                    role="user",
                    content=(
                        repair_instruction_text
                        + "\n\nAvailable numbers from cited queries (COPY EXACTLY):\n"
                        + available_str
                        + "\n\nFor numeric findings, use only the values listed above. "
                        "Copy the value exactly as shown (including all decimal places). "
                        "Do not round, compute, or substitute any other value."
                    ),
                ),
            ]
            try:
                retry = client.complete(
                    LLMRequest(
                        system=_SYSTEM,
                        messages=tuple(repair),
                        json_mode=True,
                        max_tokens=settings.llm_max_tokens,
                    )
                )
            except LLMError as error:
                return AgentResult(
                    status="error",
                    error=f"llm_error: {error.kind}: {error.message}",
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                )

            tokens_in += retry.tokens_in
            tokens_out += retry.tokens_out
            try:
                answer = parse_final_answer(
                    extract_json(retry.text), evidence=evidence, plan=plan
                )
            except ReplyInvalid as second:
                return AgentResult(
                    status="error",
                    error="answer_invalid: " + "; ".join(second.problems),
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                )

        return AgentResult(
            status="ok",
            text=render_answer(answer),
            data=to_dict(answer),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    return reporter


# --------------------------------------------------------------------------
# The answer payload
# --------------------------------------------------------------------------


def parse_final_answer(
    payload: Any, *, evidence: dict[str, Any], plan: dict[str, Any]
) -> FinalAnswer:
    """Validate the answer against the investigation it claims to summarise."""
    if not isinstance(payload, dict):
        raise AnswerInvalid(["the reply is not a JSON object"])

    problems: list[str] = []
    headline = _headline(payload.get("headline"), problems)
    prose = _prose(payload.get("answer"), problems)

    citable = _citable_queries(evidence)
    by_query = _numbers_by_query(evidence)
    findings = _findings(payload.get("findings"), citable, by_query, problems)

    assumptions = _strings(payload.get("assumptions"), "assumptions", problems)
    limitations = _strings(payload.get("limitations"), "limitations", problems)
    if not limitations:
        problems.append(
            "limitations must name at least one specific limitation of this answer"
        )

    confidence = _confidence(payload.get("confidence"), problems)
    _check_confidence(confidence, evidence, plan, findings, problems)

    if problems:
        raise AnswerInvalid(problems)
    return FinalAnswer(
        headline=headline,
        answer=prose,
        findings=tuple(findings),
        assumptions=tuple(assumptions),
        limitations=tuple(limitations),
        confidence=confidence,
    )


def _citable_queries(evidence: dict[str, Any]) -> set[str]:
    queries = evidence.get("queries")
    if not isinstance(queries, list):
        return set()
    return {
        str(record.get("id"))
        for record in queries
        if isinstance(record, dict) and record.get("status") == "ok"
    }


def _numbers_by_query(evidence: dict[str, Any]) -> dict[str, tuple[float, ...]]:
    """Which numbers the evidence can support for each query id."""
    index: dict[str, list[float]] = {}
    findings = evidence.get("findings")
    if not isinstance(findings, list):
        return {}
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        numbers = finding.get("numbers")
        if not isinstance(numbers, dict):
            continue
        values = [
            float(value)
            for value in numbers.values()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ]
        for query_id in finding.get("evidence_query_ids") or []:
            index.setdefault(str(query_id), []).extend(values)
    return {query_id: tuple(values) for query_id, values in index.items()}


def _findings(
    value: Any,
    citable: set[str],
    by_query: dict[str, tuple[float, ...]],
    problems: list[str],
) -> list[AnswerFinding]:
    if value is None:
        return []
    if not isinstance(value, list):
        problems.append("findings must be a list")
        return []
    if len(value) > _FINDINGS_MAX:
        problems.append(f"findings has more than {_FINDINGS_MAX} entries")

    findings: list[AnswerFinding] = []
    for index, raw in enumerate(value):
        where = f"findings[{index}]"
        if not isinstance(raw, dict):
            problems.append(f"{where} must be an object")
            continue

        claim = raw.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            problems.append(f"{where}.claim must be a non-empty string")
            claim = ""
        elif len(" ".join(claim.split())) > _CLAIM_LIMIT:
            problems.append(f"{where}.claim is longer than {_CLAIM_LIMIT} characters")
        claim = " ".join(claim.split())

        cited = _citations(raw.get("evidence_query_ids"), where, citable, problems)
        numbers = _numbers(raw.get("numbers"), where, cited, by_query, problems)
        findings.append(
            AnswerFinding(
                claim=claim,
                evidence_query_ids=tuple(cited),
                numbers=numbers,
            )
        )
    return findings


def _citations(
    value: Any, where: str, citable: set[str], problems: list[str]
) -> list[str]:
    if not isinstance(value, list) or not value:
        problems.append(f"{where}.evidence_query_ids must list at least one query id")
        return []
    cited: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            problems.append(f"{where}.evidence_query_ids entries must be strings")
            continue
        identifier = entry.strip()
        if identifier not in citable:
            problems.append(
                f"{where} cites {identifier!r}, which is not a query that returned "
                "rows in the investigation"
            )
            continue
        cited.append(identifier)
    return cited


def _numbers(
    value: Any,
    where: str,
    cited: list[str],
    by_query: dict[str, tuple[float, ...]],
    problems: list[str],
) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        problems.append(f"{where}.numbers must be an object of name -> number")
        return {}
    if len(value) > _MAX_NUMBERS:
        problems.append(f"{where}.numbers has more than {_MAX_NUMBERS} entries")

    allowed = [number for query_id in cited for number in by_query.get(query_id, ())]
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
        if not any(_same_number_exact(float(raw), number) for number in allowed):
            available = ", ".join(f"{number:.10g}" for number in sorted(set(allowed)))
            problems.append(
                f"{where}.numbers[{key!r}]={float(raw):.10g} is not in the evidence "
                f"cited by {', '.join(cited) or 'no query'}; available: "
                f"{available or 'none'}"
            )
            continue
        numbers[key] = float(raw)
    return numbers


def _same_number_exact(left: float, right: float) -> bool:
    """Compare numbers exactly, preserving full precision from investigation.
    
    Unlike _same_number (which uses 9g for tolerance), this requires exact match
    to ensure the reporter copies numbers exactly from investigation findings
    without rounding or computation (per D-22).
    """
    return f"{left:.10g}" == f"{right:.10g}"


def _same_number(left: float, right: float) -> bool:
    """Compare magnitudes, not signs and not float noise.

    The direction of a change belongs to the prose ("fell 8.03%"); the number is a
    magnitude either way. Nine significant digits absorb JSON round-tripping while
    still refusing a rounded or reshaped figure.
    """
    return f"{abs(left):.9g}" == f"{abs(right):.9g}"


def _check_confidence(
    confidence: str,
    evidence: dict[str, Any],
    plan: dict[str, Any],
    findings: list[AnswerFinding],
    problems: list[str],
) -> None:
    if confidence not in _CONFIDENCE:
        return  # already reported by _confidence
    stop_reason = evidence.get("stop_reason")
    evidence_findings = evidence.get("findings")
    count = len(evidence_findings) if isinstance(evidence_findings, list) else 0

    if stop_reason and stop_reason != "model_finished" and confidence == "high":
        problems.append(
            f"confidence 'high' is not supported: the investigation stopped with "
            f"{stop_reason}. Use 'medium' or 'low' instead."
        )
    if plan.get("answerable") is False and confidence != "low":
        problems.append(
            "confidence must be 'low': the plan reports that this schema cannot "
            "answer the question"
        )
    if count == 0 and confidence != "low":
        problems.append(
            "confidence must be 'low': the investigation returned no findings"
        )
    if count > 0 and not findings:
        problems.append(
            f"the investigation produced {count} finding(s), so the answer must "
            "present at least one of them with its query ids"
        )


def _headline(value: Any, problems: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        problems.append("headline must be a non-empty string")
        return ""
    headline = " ".join(value.split())
    if len(headline) > _HEADLINE_LIMIT:
        problems.append(f"headline is longer than {_HEADLINE_LIMIT} characters")
        headline = headline[:_HEADLINE_LIMIT]
    return headline


def _prose(value: Any, problems: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        problems.append("answer must be a non-empty string")
        return ""
    prose = value.strip()
    if len(prose) > _ANSWER_LIMIT:
        problems.append(f"answer is longer than {_ANSWER_LIMIT} characters")
        prose = prose[:_ANSWER_LIMIT]
    return prose


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


def _confidence(value: Any, problems: list[str]) -> str:
    """Parse and validate the confidence field.
    
    Returns the confidence value or None if invalid, letting the caller
    handle the problem consistently.
    """
    if not isinstance(value, str):
        problems.append(
            "confidence must be one of " + ", ".join(repr(item) for item in _CONFIDENCE)
        )
        return "medium"
    
    # Extract the first word (handles "high confidence", "HIGH", etc.)
    words = value.strip().lower().split()
    if not words:
        problems.append(
            "confidence must be one of " + ", ".join(repr(item) for item in _CONFIDENCE)
        )
        return "medium"
    
    confidence = words[0]
    if confidence not in _CONFIDENCE:
        problems.append(
            "confidence must be one of " + ", ".join(repr(item) for item in _CONFIDENCE)
        )
        return "medium"
    return confidence


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_answer(answer: FinalAnswer) -> str:
    """The answer artifact: what a reader sees, and what the CLI prints."""
    lines = [answer.headline, f"confidence: {answer.confidence}", "", answer.answer]

    if answer.findings:
        lines += ["", f"findings: {len(answer.findings)}"]
        for finding in answer.findings:
            cited = ", ".join(finding.evidence_query_ids) or "no evidence"
            lines.append(f"  [{cited}]  {finding.claim}")
            if finding.numbers:
                lines.append(
                    "      numbers: "
                    + ", ".join(
                        f"{name}={_number(value)}"
                        for name, value in finding.numbers.items()
                    )
                )
    if answer.assumptions:
        lines += ["", "assumptions:"] + [f"  - {item}" for item in answer.assumptions]
    if answer.limitations:
        lines += ["", "limitations:"] + [f"  - {item}" for item in answer.limitations]
    return "\n".join(lines)


def _number(value: float) -> str:
    return f"{value:.10g}"
