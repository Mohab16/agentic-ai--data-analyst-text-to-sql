"""Context is built from declared reads and bounded (X-01, X-02, X-03)."""

from __future__ import annotations

from core.context import TRUNCATION_MARKER, build_context, truncate
from core.types import AgentResult, Step


def step(reads: tuple[str, ...], instruction: str = "do the thing") -> Step:
    return Step(id="s", agent="planner", instruction=instruction, reads=reads)


def ok(text: str, data: dict[str, object] | None = None) -> AgentResult:
    return AgentResult(status="ok", text=text, data=data)


def test_only_declared_reads_are_included() -> None:
    artifacts = {
        "question": ok("why did revenue drop?"),
        "schema": ok("tables: orders"),
        "unrelated": ok("SECRET"),
    }
    context, truncated = build_context(step(("question", "schema")), artifacts, 10_000)
    assert "why did revenue drop?" in context
    assert "tables: orders" in context
    assert "SECRET" not in context
    assert truncated is False


def test_instruction_is_always_first() -> None:
    context, _ = build_context(
        step(("question",), instruction="INSTRUCTION-TEXT"),
        {"question": ok("payload")},
        10_000,
    )
    assert context.index("INSTRUCTION-TEXT") < context.index("payload")


def test_artifact_ids_are_visible_for_debugging() -> None:
    context, _ = build_context(step(("question",)), {"question": ok("x")}, 10_000)
    assert "[artifact: question]" in context


def test_failed_artifacts_are_not_fed_to_the_next_step() -> None:
    artifacts = {"plan": AgentResult(status="error", error="planner_failed")}
    context, _ = build_context(step(("plan",)), artifacts, 10_000)
    assert "planner_failed" not in context


def test_missing_artifact_is_skipped() -> None:
    context, truncated = build_context(step(("ghost",)), {}, 10_000)
    assert "ghost" not in context
    assert truncated is False


def test_structured_payload_is_rendered() -> None:
    artifacts = {"plan": ok("", {"metric": "revenue"})}
    context, _ = build_context(step(("plan",)), artifacts, 10_000)
    assert "[structured]" in context
    assert '"metric": "revenue"' in context


def test_budget_is_never_exceeded() -> None:
    artifacts = {"question": ok("x" * 50_000)}
    context, truncated = build_context(step(("question",)), artifacts, 2_000)
    assert truncated is True
    assert len(context) <= 2_000


def test_truncation_is_announced() -> None:
    context, truncated = build_context(step(("question",)), {"question": ok("y" * 5_000)}, 1_000)
    assert truncated is True
    marker_head = TRUNCATION_MARKER.split("{omitted}")[0]
    assert marker_head in context


def test_truncation_keeps_the_head_and_the_tail() -> None:
    text = "HEAD-MARKER" + ("m" * 5_000) + "TAIL-MARKER"
    context, _ = build_context(step(("question",)), {"question": ok(text)}, 1_000)
    assert "HEAD-MARKER" in context
    assert "TAIL-MARKER" in context


def test_truncation_is_deterministic() -> None:
    artifacts = {"question": ok("z" * 3_000)}
    first = build_context(step(("question",)), artifacts, 800)
    second = build_context(step(("question",)), artifacts, 800)
    assert first == second


def test_no_truncation_when_within_budget() -> None:
    context, truncated = truncate("short", 100)
    assert context == "short"
    assert truncated is False


def test_tiny_budget_still_respects_the_limit() -> None:
    context, truncated = truncate("q" * 5_000, 10)
    assert truncated is True
    assert len(context) <= 10


def test_zero_budget_returns_empty_and_flagged() -> None:
    context, truncated = truncate("anything", 0)
    assert context == ""
    assert truncated is True
