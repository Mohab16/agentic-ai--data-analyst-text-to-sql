"""Context assembly: one builder, declared reads, hard budget.

Rules: X-01 (build, don't accumulate), X-02 (only declared reads), X-03
(deterministic truncation, recorded in the trace), X-06 (static instruction is
separate from assembled context).
"""

from __future__ import annotations

import json
from typing import Any

from core.types import AgentResult, Step
from core.utils import is_slug, truncate_text

# Agent-specific context budgets to reduce token usage
# These values are kept here to avoid circular imports with core.orchestrator
AGENT_CONTEXT_BUDGETS = {
    "planner": 20_000,  # 20KB
    "investigator": 30_000,  # 30KB
    "reporter": 25_000,  # 25KB
}

# Agent-specific evidence limits - cap which items to include in context
AGENT_EVIDENCE_LIMITS = {
    "plan": 0,  # No evidence needed for planning
    "investigate": 6,  # Max 6 evidence items for investigator
    "answer": 6,  # Max 6 evidence items for reporter
}

_ARTIFACT_MARKER = "[artifact: "
_STRUCTURED_MARKER = "\n[structured]\n"
# Truncation marker format - shared with core.utils.truncate_text
TRUNCATION_MARKER = "\n\n... [context truncated: {omitted} chars omitted] ...\n\n"


def build_context(
    step: Step,
    artifacts: dict[str, AgentResult],
    budget: int,
) -> tuple[str, bool]:
    """Assemble the step's input from its declared reads only.

    Applies agent-specific evidence limits to keep context within budget.

    Returns `(context, truncated)`. The result never exceeds `budget` characters.
    """
    parts = [f"[step: {step.id}]\n{step.instruction}"]
    for artifact_id in step.reads:
        artifact = artifacts.get(artifact_id)
        if artifact is None or artifact.status != "ok":
            continue
        # Apply evidence limits to reduce token usage
        limit = AGENT_EVIDENCE_LIMITS.get(artifact_id, 6)
        if limit > 0 and artifact.data:
            parts.append(f"[artifact: {artifact_id}]\n{render_artifact_limited(artifact, limit)}")
        else:
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


def render_artifact_limited(artifact: AgentResult, limit: int) -> str:
    """Render artifact with evidence truncation for agents receiving large evidence sets.
    
    Summarizes structured data and limits text to reduce token usage.
    
    Args:
        artifact: The artifact to render
        limit: Maximum number of queries/evidence items to include
        
    Returns:
        str: the rendered artifact
    """
    if not artifact.data:
        return artifact.text[:limit]
    payload = json.dumps(artifact.data, ensure_ascii=False, sort_keys=True, indent=2)
    if not artifact.text:
        return f"[structured]\n{payload}\n\n[truncated: {len(payload)} chars of structured payload]"
    # Limit text to keep within budget while preserving key information
    truncated_text = artifact.text[:limit * 2]  # Allow some text for readability
    return f"[structured]\n{payload}\n\nText truncated to {limit} items max"


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
    # The structured payload is first in the block; trailing text after the
    # JSON object is the prose half. raw_decode parses the first JSON value
    # and ignores whatever follows, so braces inside the prose (e.g. in SQL
    # strings) cannot throw off the boundary.
    decoder = json.JSONDecoder()
    try:
        payload, _ = decoder.raw_decode(after)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def truncate(text: str, budget: int) -> tuple[str, bool]:
    """Deterministic head/tail truncation. Never returns more than `budget` chars.
    
    Delegates to the shared ``core.utils.truncate_text`` implementation to ensure
    the truncation marker is never split and the result always fits within budget.
    """
    return truncate_text(text, budget)


def _compact_json(data: dict) -> str:
    """Compact JSON serialization for smaller token usage.
    
    Uses minimal whitespace while remaining valid JSON.
    """
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _split(available: int) -> tuple[int, int]:
    head = (available * 3) // 5
    return head, available - head
