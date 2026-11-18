"""The step loop: dispatch, budgets, state transitions, trace emission (O-02).

Fake agents are `AgentResult` factories, not mocks with prompt assertions.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from core.config import Settings
from core.orchestrator import (
    AGENT_KEYS,
    ROOT_ARTIFACTS,
    AgentFn,
    build_plan,
    execute,
    validate_plan,
)
from core.types import AgentResult, Plan, PlanInvalid, Step, TraceEvent


class RecordingTrace:
    """A three-line stand-in for `TraceStore` (R-01 is what we assert)."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def emit(self, event: TraceEvent) -> None:
        self.events.append(event)


def roots(question: str = "why did revenue drop?") -> dict[str, AgentResult]:
    return {
        "question": AgentResult(status="ok", text=question),
        "schema": AgentResult(status="ok", text="tables: orders, order_items"),
    }


def agents(
    planner: AgentFn | None = None,
    investigator: AgentFn | None = None,
    reporter: AgentFn | None = None,
) -> dict[str, AgentFn]:
    return {
        "planner": planner or (lambda ctx, s: AgentResult(status="ok", text="planned")),
        "investigator": investigator
        or (lambda ctx, s: AgentResult(status="ok", text="investigated")),
        "reporter": reporter or (lambda ctx, s: AgentResult(status="ok", text="answered")),
    }


def test_happy_path_completes_every_step(settings: Settings) -> None:
    trace = RecordingTrace()
    outcome = execute(build_plan("tsk_1"), agents(), settings, trace, roots())

    assert outcome.state.status == "done"
    assert outcome.state.steps_completed == 3
    assert outcome.state.current_step is None
    assert set(outcome.artifacts) == {"question", "schema", "plan", "investigate", "answer"}


def test_one_trace_event_per_step_in_order(settings: Settings) -> None:
    trace = RecordingTrace()
    execute(build_plan("tsk_1"), agents(), settings, trace, roots())

    assert [event.step_id for event in trace.events] == ["plan", "investigate", "answer"]
    assert [event.agent for event in trace.events] == list(AGENT_KEYS)
    assert all(event.task_id == "tsk_1" for event in trace.events)
    assert all(event.model == settings.llm_model for event in trace.events)


def test_agent_sees_its_declared_reads_only(settings: Settings) -> None:
    seen: list[str] = []

    def planner(ctx: str, _settings: Settings) -> AgentResult:
        seen.append(ctx)
        return AgentResult(status="ok", text="plan-text")

    execute(build_plan("tsk_1"), agents(planner=planner), settings, RecordingTrace(), roots())

    assert "why did revenue drop?" in seen[0]
    assert "tables: orders" in seen[0]
    assert "plan-text" not in seen[0]


def test_later_step_receives_earlier_artifacts(settings: Settings) -> None:
    seen: list[str] = []

    def reporter(ctx: str, _settings: Settings) -> AgentResult:
        seen.append(ctx)
        return AgentResult(status="ok", text="ok")

    execute(build_plan("tsk_1"), agents(reporter=reporter), settings, RecordingTrace(), roots())

    assert "planned" in seen[0]
    assert "investigated" in seen[0]


def test_agent_error_stops_the_workflow(settings: Settings) -> None:
    trace = RecordingTrace()
    outcome = execute(
        build_plan("tsk_1"),
        agents(investigator=lambda ctx, s: AgentResult(status="error", error="model_call_failed")),
        settings,
        trace,
        roots(),
    )

    assert outcome.state.status == "error"
    assert outcome.state.error == "model_call_failed"
    assert outcome.state.steps_completed == 2
    assert "answer" not in outcome.artifacts
    assert [event.step_id for event in trace.events] == ["plan", "investigate"]
    assert trace.events[-1].outcome == "error"


def test_raising_agent_becomes_a_typed_error(settings: Settings) -> None:
    def explode(ctx: str, _settings: Settings) -> AgentResult:
        raise RuntimeError("boom")

    outcome = execute(
        build_plan("tsk_1"), agents(planner=explode), settings, RecordingTrace(), roots()
    )

    assert outcome.state.status == "error"
    assert outcome.state.error is not None
    assert "agent_raised" in outcome.state.error
    assert "RuntimeError" in outcome.state.error


def test_no_error_text_leaks_into_artifacts_as_success(settings: Settings) -> None:
    outcome = execute(
        build_plan("tsk_1"),
        agents(investigator=lambda ctx, s: AgentResult(status="error", error="upstream_500")),
        settings,
        RecordingTrace(),
        roots(),
    )

    failed = outcome.artifacts["investigate"]
    assert failed.status == "error"
    assert failed.text == ""
    assert outcome.state.status == "error"


def test_wall_clock_budget_stops_before_the_first_step(settings: Settings) -> None:
    trace = RecordingTrace()
    outcome = execute(
        build_plan("tsk_1"),
        agents(),
        replace(settings, wall_clock_s=-1),
        trace,
        roots(),
    )

    assert outcome.state.status == "error"
    assert outcome.state.error is not None
    assert "wall_clock" in outcome.state.error
    assert trace.events == []
    assert outcome.state.steps_completed == 0


def test_truncated_context_is_recorded_in_the_trace(settings: Settings) -> None:
    trace = RecordingTrace()
    execute(
        build_plan("tsk_1"),
        agents(),
        replace(settings, context_budget=120),
        trace,
        roots(question="q" * 5_000),
    )

    assert trace.events[0].outcome == "truncated"


def test_unknown_agent_is_rejected_before_execution(settings: Settings) -> None:
    plan = Plan(task_id="t", steps=(Step(id="a", agent="ghost", instruction="x"),))
    with pytest.raises(PlanInvalid):
        execute(plan, agents(), settings, RecordingTrace(), roots())


def test_roots_are_available_without_a_step(settings: Settings) -> None:
    outcome = execute(build_plan("tsk_1"), agents(), settings, RecordingTrace(), roots("Q"))
    assert outcome.artifacts["question"].text == "Q"


def test_built_plan_routes_to_known_agents(settings: Settings) -> None:
    strict = {key: agents()[key] for key in AGENT_KEYS}
    outcome = execute(build_plan("tsk_1"), strict, settings, RecordingTrace(), roots())
    assert outcome.state.status == "done"


def test_state_is_never_left_running(settings: Settings) -> None:
    for supplied in (agents(), agents(reporter=lambda c, s: AgentResult(status="error", error="x"))):
        outcome = execute(build_plan("tsk_1"), supplied, settings, RecordingTrace(), roots())
        assert outcome.state.status in ("done", "error")


def test_root_artifact_names_are_declared() -> None:
    assert ROOT_ARTIFACTS == ("question", "schema")
    validate_plan(build_plan("tsk_1"), AGENT_KEYS, ROOT_ARTIFACTS)
