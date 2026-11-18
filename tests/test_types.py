"""Contracts must be JSON-serializable and losslessly reconstructible (C-01)."""

from __future__ import annotations

import json

from core.types import (
    AgentResult,
    AnswerFinding,
    Check,
    FinalAnswer,
    Finding,
    InvestigationPlan,
    InvestigationReport,
    Plan,
    QueryRecord,
    Step,
    TaskState,
    Timeframe,
    ToolError,
    TraceEvent,
    from_dict,
    new_task_id,
    to_dict,
)


def roundtrip[ContractT](contract: ContractT) -> ContractT:
    payload = json.loads(json.dumps(to_dict(contract)))
    return from_dict(type(contract), payload)


def test_step_roundtrip() -> None:
    step = Step(id="plan", agent="planner", instruction="do it", reads=("a", "b"))
    assert roundtrip(step) == step


def test_step_reads_is_a_tuple_after_roundtrip() -> None:
    rebuilt = roundtrip(Step(id="x", agent="y", instruction="z", reads=("a",)))
    assert isinstance(rebuilt.reads, tuple)


def test_plan_roundtrip() -> None:
    plan = Plan(
        task_id="tsk_1",
        steps=(Step(id="plan", agent="planner", instruction="i"),),
        max_steps=5,
    )
    assert roundtrip(plan) == plan


def test_agent_result_roundtrip_with_payload() -> None:
    result = AgentResult(
        status="ok",
        text="done",
        data={"nested": {"x": 1}, "items": [1, 2]},
        tool_calls=("execute_sql",),
        tokens_in=10,
        tokens_out=20,
    )
    assert roundtrip(result) == result


def test_agent_result_error_has_no_text_error_mixing() -> None:
    result = AgentResult(status="error", error="model_call_failed: 429")
    rebuilt = roundtrip(result)
    assert rebuilt.status == "error"
    assert rebuilt.text == ""
    assert rebuilt.error == "model_call_failed: 429"


def test_task_state_roundtrip() -> None:
    state = TaskState(
        task_id="tsk_1", status="error", current_step="plan", error="boom", steps_completed=1
    )
    assert roundtrip(state) == state


def test_trace_event_roundtrip() -> None:
    event = TraceEvent(
        task_id="tsk_1",
        step_id="plan",
        agent="planner",
        model="m",
        started_at="2026-01-01T00:00:00+00:00",
        duration_ms=12,
        outcome="truncated",
        tool_calls=("a",),
        error=None,
    )
    assert roundtrip(event) == event


def test_tool_error_roundtrip() -> None:
    error = ToolError(
        tool="execute_sql",
        kind="invalid_input",
        message="bad",
        retryable=False,
        error_id="err_001",
    )
    rebuilt = roundtrip(error)
    assert rebuilt == error
    assert rebuilt.error_id == "err_001"


def test_investigation_plan_roundtrip() -> None:
    plan = InvestigationPlan(
        metric_definition="revenue = sum(order_items.line_total)",
        timeframe=Timeframe(start="2026-10-01", end="2026-10-31", comparison="prior_month"),
        answerable=True,
        checks=(
            Check(
                id="c1",
                question="did revenue drop vs prior month?",
                method="sum by month",
                dimensions=("month",),
            ),
        ),
        assumptions=("revenue excludes returns",),
    )
    assert roundtrip(plan) == plan


def test_investigation_plan_preserves_check_tuple() -> None:
    rebuilt = roundtrip(
        InvestigationPlan(
            metric_definition="m",
            timeframe=Timeframe(start="a", end="b", comparison="c"),
            answerable=True,
            checks=(Check(id="c1", question="q", method="m"),),
        )
    )
    assert isinstance(rebuilt.checks, tuple)
    assert isinstance(rebuilt.checks[0], Check)


def test_investigation_report_roundtrip() -> None:
    report = InvestigationReport(
        findings=(
            Finding(
                id="f1",
                claim="revenue fell 18%",
                evidence_query_ids=("q1", "q2"),
                numbers={"revenue_now": 1000.0, "revenue_before": 1219.5},
            ),
        ),
        queries=(
            QueryRecord(
                id="q1",
                sql="select 1",
                status="ok",
                row_count=1,
                duration_ms=3,
                columns=("a",),
                preview="a\n1",
            ),
            QueryRecord(
                id="q2",
                sql="select 2",
                status="error",
                error_kind="invalid_input",
                error_message="no such column",
            ),
        ),
        stop_reason="budget_exhausted",
    )
    assert roundtrip(report) == report


def test_final_answer_roundtrip() -> None:
    answer = FinalAnswer(
        headline="Revenue fell 18%",
        answer="Long prose.",
        findings=(AnswerFinding(claim="c", evidence_query_ids=("q1",), numbers={"pct": -18.0}),),
        assumptions=("month = calendar month",),
        limitations=("correlation only",),
        confidence="medium",
    )
    assert roundtrip(answer) == answer


def test_numbers_dict_survives_roundtrip() -> None:
    rebuilt = roundtrip(
        Finding(id="f", claim="c", numbers={"a": 1.5, "b": -2.0})
    )
    assert rebuilt.numbers == {"a": 1.5, "b": -2.0}


def test_to_dict_converts_dates() -> None:
    from datetime import date

    assert to_dict(date(2026, 1, 2)) == "2026-01-02"


def test_task_ids_are_unique_and_prefixed() -> None:
    ids = {new_task_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(task_id.startswith("tsk_") for task_id in ids)
