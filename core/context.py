"""Context assembly: one builder, declared reads, hard budget.

Rules: X-01 (build, don't accumulate), X-02 (only declared reads), X-03
(deterministic truncation, recorded in the trace), X-06 (static instruction is
separate from assembled context).
"""

from __future__ import annotations

import json
from typing import Any

from core.types import AgentResult, Step

TRUNCATION_MARKER = "\n\n... [context truncated: {omitted} chars omitted] ...\n\n"

_HEAD_SHARE = 3  # of 5: instruction and early evidence survive

_ARTIFACT_MARKER = "[artifact: "
_STRUCTURED_MARKER = "\n[structured]\n"


def build_context(
    step: Step,
    artifacts: dict[str, AgentResult],
    budget: int,
) -> tuple[str, bool]:
    """Assemble the step's input from its declared reads only.

    Returns `(context, truncated)`. The result never exceeds `budget` characters.
    """
    parts = [f"[step: {step.id}]\n{step.instruction}"]
    for artifact_id in step.reads:
        artifact = artifacts.get(artifact_id)
        if artifact is None or artifact.status != "ok":
            continue
        parts.append(f"[artifact: {artifact_id}]\n{render_artifact(artifact)}")
    return truncate("\n\n".join(parts), budget)


def render_artifact(artifact: AgentResult) -> str:
    """Structured payload first, text second. Truncation may cut the text.

    The structured half is what downstream agents read back with `read_artifact`,
    so it goes first and the head-heavy truncation keeps it intact.
    """
    if not artifact.data:
        return artifact.text
    payload = json.dumps(artifact.data, ensure_ascii=False, sort_keys=True, indent=2)
    if not artifact.text:
        return f"[structured]\n{payload}"
    return f"[structured]\n{payload}\n\n{artifact.text}"


def read_artifact(context: str, artifact_id: str) -> dict[str, Any] | None:
    """The structured payload of a declared artifact, as the agent received it.

    Agents have to check their own output against upstream evidence, and the
    evidence reaches them as text. The `[structured]` block is the machine-readable
    half of that text, so this is the supported way to read it back: no agent
    hand-parses the rendering, and no agent reaches past the context it was given
    (X-02).

    Returns `None` when the artifact is absent, carries no payload, or its payload
    was cut by truncation. Callers must fail closed on `None` rather than proceeding
    without the evidence they meant to check against.
    """
    header = f"{_ARTIFACT_MARKER}{artifact_id}]"
    start = context.find(header)
    if start < 0:
        return None

    body_start = start + len(header)
    next_start = context.find(f"\n\n{_ARTIFACT_MARKER}", body_start)
    block = context[body_start:next_start] if next_start >= 0 else context[body_start:]

    marker = block.find(_STRUCTURED_MARKER)
    if marker < 0:
        return None

    after = block[marker + len(_STRUCTURED_MARKER) :]
    try:
        payload = json.loads(after)
    except json.JSONDecodeError:
        # The structured payload is first in the block; trailing text after the
        # JSON object is the prose half. Slice from the first { to the last }.
        first = after.find("{")
        last = after.rfind("}")
        if first < 0 or last < first:
            return None
        try:
            payload = json.loads(after[first : last + 1])
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def truncate(text: str, budget: int) -> tuple[str, bool]:
    """Deterministic head/tail truncation. Never returns more than `budget` chars."""
    if budget <= 0:
        return "", True
    if len(text) <= budget:
        return text, False

    marker = TRUNCATION_MARKER.format(omitted=len(text))
    for _ in range(4):
        head, tail = _split(max(0, budget - len(marker)))
        candidate = TRUNCATION_MARKER.format(omitted=len(text) - head - tail)
        if len(candidate) == len(marker):
            marker = candidate
            break
        marker = candidate

    head, tail = _split(max(0, budget - len(marker)))
    head_text = text[:head]
    tail_text = text[-tail:] if tail > 0 else ""
    result = f"{head_text}{marker}{tail_text}"
    return result[:budget], True


def _split(available: int) -> tuple[int, int]:
    head = (available * _HEAD_SHARE) // 5
    return head, available - head
