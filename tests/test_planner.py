"""The planner: prompt shape, payload validation, one repair turn, failure kinds.

The model is a `ScriptedLLM` here, so these tests are about the contract between
the model and the code — exactly the part that must not be trusted to the model.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agents.planner import (
    PlanPayloadInvalid,
    make_planner,
    parse_plan_payload,
    render_plan_summary,
)
from core.config import Settings
from core.types import InvestigationPlan, from_dict
from llm.base import LLMError

CONTEXT = "[step: plan]\nPlan it.\n\n[artifact: schema]\norders (1 row)"


def valid_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "metric_definition": (
            "revenue = SUM(order_items.line_total) for orders with status 'completed'"
        ),
        "metric_type": "gross",
        "timeframe": {
            "start": "2026-10-01",
            "end": "2026-10-31",
            "comparison": "previous month",
            "granularity": "month",
        },
        "answerable": True,
        "checks": [
            {
                "id": "c1",
                "question": "How did total revenue change month over month?",
                "method": "sum line_total by month, compare with the baseline",
                "dimensions": ["month"],
            },
            {
                "id": "c2",
                "question": "Which regions contributed most of the change?",
                "method": "sum line_total by region for both months",
                "dimensions": ["region"],
            },
        ],
        "blocking_gap": None,
        "assumptions": ["completed orders only"],
    }
    payload.update(overrides)
    return payload


def plan(settings: Settings, scripted_llm, *replies: object):
    client = scripted_llm(list(replies))
    return make_planner(client)(CONTEXT, settings), client


def problems_for(payload: Any, *, max_checks: int = 8) -> list[str]:
    with pytest.raises(PlanPayloadInvalid) as caught:
        parse_plan_payload(payload, max_checks=max_checks)
    return list(caught.value.problems)


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_a_valid_plan_becomes_an_investigation_plan() -> None:
    plan = parse_plan_payload(valid_payload(), max_checks=8)

    assert isinstance(plan, InvestigationPlan)
    assert plan.answerable is True
    assert plan.timeframe.start == "2026-10-01"
    assert plan.timeframe.comparison == "previous month"
    assert plan.timeframe.granularity == "month"
    assert [check.id for check in plan.checks] == ["c1", "c2"]
    assert plan.checks[1].dimensions == ("region",)
    assert plan.blocking_gap is None
    assert plan.assumptions == ("completed orders only",)


def test_the_agent_result_carries_structured_and_readable_forms(
    settings: Settings, scripted_llm
) -> None:
    result, _client = plan(settings, scripted_llm, json.dumps(valid_payload()))

    assert result.status == "ok"
    assert result.error is None
    assert result.data is not None
    restored = from_dict(InvestigationPlan, result.data)
    assert restored.metric_definition.startswith("revenue = SUM")
    assert "metric:" in result.text
    assert "checks: 2" in result.text
    assert "assumptions:" in result.text


def test_the_context_is_the_user_message_and_the_instruction_is_static(
    settings: Settings, scripted_llm
) -> None:
    _result, client = plan(settings, scripted_llm, json.dumps(valid_payload()))

    request = client.requests[0]
    assert request.system.startswith("You are the planner")
    assert "one JSON object" in request.system
    assert [message.role for message in request.messages] == ["user"]
    assert request.messages[0].content == CONTEXT
    assert request.json_mode is True
    assert request.max_tokens == settings.llm_max_tokens


def test_the_instruction_states_the_check_limit(settings: Settings, scripted_llm) -> None:
    _result, client = plan(settings, scripted_llm, json.dumps(valid_payload()))
    assert f"1 and {settings.max_queries} checks" in client.requests[0].system


def test_the_instruction_forbids_sql_in_the_plan(settings: Settings, scripted_llm) -> None:
    _result, client = plan(settings, scripted_llm, json.dumps(valid_payload()))
    assert "never SQL" in client.requests[0].system


@pytest.mark.parametrize(
    "reply",
    [
        "```json\n" + json.dumps(valid_payload()) + "\n```",
        "Here is the plan you asked for:\n" + json.dumps(valid_payload()),
        json.dumps(valid_payload()) + "\nLet me know if you need anything else.",
    ],
)
def test_json_survives_fences_and_chatter(
    settings: Settings, scripted_llm, reply: str
) -> None:
    result, _client = plan(settings, scripted_llm, reply)
    assert result.status == "ok"


def test_extra_keys_from_the_model_are_ignored(
    settings: Settings, scripted_llm
) -> None:
    payload = valid_payload(confidence="high", reasoning="because")
    result, _client = plan(settings, scripted_llm, json.dumps(payload))
    assert result.status == "ok"


# --------------------------------------------------------------------------
# Repair turn
# --------------------------------------------------------------------------


def test_invalid_json_gets_exactly_one_repair(settings: Settings, scripted_llm) -> None:
    result, client = plan(
        settings, scripted_llm, "I cannot do JSON today", json.dumps(valid_payload())
    )

    assert result.status == "ok"
    assert len(client.requests) == 2
    repair = client.requests[1].messages
    assert [message.role for message in repair] == ["user", "assistant", "user"]
    assert repair[1].content == "I cannot do JSON today"
    assert "rejected by the plan validator" in repair[2].content
    assert "JSON object" in repair[2].content


def test_the_repair_turn_lists_every_problem(settings: Settings, scripted_llm) -> None:
    broken = valid_payload(metric_definition="", answerable="yes")
    result, client = plan(settings, scripted_llm, json.dumps(broken), json.dumps(valid_payload()))

    assert result.status == "ok"
    repair_text = client.requests[1].messages[-1].content
    assert "metric_definition" in repair_text
    assert "answerable" in repair_text


def test_a_second_failure_is_an_error_not_a_loop(
    settings: Settings, scripted_llm
) -> None:
    result, client = plan(settings, scripted_llm, "nope", "still nope")

    assert result.status == "error"
    assert len(client.requests) == 2
    assert result.error is not None
    assert result.error.startswith("plan_invalid:")
    assert "JSON object" in result.error


def test_an_empty_reply_is_repaired_without_an_empty_assistant_turn(
    settings: Settings, scripted_llm
) -> None:
    result, client = plan(settings, scripted_llm, "   ", json.dumps(valid_payload()))

    assert result.status == "ok"
    assert [message.role for message in client.requests[1].messages] == ["user", "user"]


def test_tokens_from_both_attempts_are_counted(settings: Settings, scripted_llm) -> None:
    result, _client = plan(
        settings, scripted_llm, "nope", json.dumps(valid_payload())
    )
    assert (result.tokens_in, result.tokens_out) == (20, 10)


def test_a_model_failure_becomes_an_agent_error(settings: Settings, scripted_llm) -> None:
    result, client = plan(
        settings, scripted_llm, LLMError("rate_limit", "slow down", retryable=True)
    )

    assert result.status == "error"
    assert result.error == "llm_error: rate_limit: slow down"
    assert len(client.requests) == 1  # the client owns transport retries, not the agent


# --------------------------------------------------------------------------
# Payload validation
# --------------------------------------------------------------------------


def test_missing_metric_is_reported() -> None:
    payload = valid_payload()
    del payload["metric_definition"]
    assert any("metric_definition" in problem for problem in problems_for(payload))


def test_missing_timeframe_is_reported() -> None:
    payload = valid_payload()
    del payload["timeframe"]
    assert any("timeframe must be an object" in p for p in problems_for(payload))


@pytest.mark.parametrize("bad", ["2026-02-30", "2026-13-01", "yesterday", ""])
def test_impossible_dates_are_rejected(bad: str) -> None:
    payload = valid_payload()
    payload["timeframe"]["start"] = bad
    assert any("timeframe.start" in problem for problem in problems_for(payload))


def test_start_after_end_is_rejected() -> None:
    payload = valid_payload()
    payload["timeframe"].update({"start": "2026-11-01", "end": "2026-10-01"})
    assert any("after" in problem for problem in problems_for(payload))


def test_unknown_granularity_is_rejected() -> None:
    payload = valid_payload()
    payload["timeframe"]["granularity"] = "quarter"
    assert any("granularity" in problem for problem in problems_for(payload))


def test_granularity_defaults_to_day() -> None:
    payload = valid_payload()
    del payload["timeframe"]["granularity"]
    assert parse_plan_payload(payload, max_checks=8).timeframe.granularity == "day"


def test_answerable_must_be_a_boolean() -> None:
    payload = valid_payload(answerable="yes")
    assert any("true or false" in problem for problem in problems_for(payload))


def test_an_answerable_plan_needs_at_least_one_check() -> None:
    payload = valid_payload(checks=[])
    assert any("at least one check" in p for p in problems_for(payload))


def test_too_many_checks_are_rejected() -> None:
    payload = valid_payload(
        checks=[
            {"id": f"c{index}", "question": "q", "method": "m"}
            for index in range(1, 10)
        ]
    )
    assert any("at most 8" in problem for problem in problems_for(payload))


def test_duplicate_check_ids_are_rejected() -> None:
    payload = valid_payload(
        checks=[
            {"id": "c1", "question": "a", "method": "m"},
            {"id": "c1", "question": "b", "method": "m"},
        ]
    )
    assert any("duplicated" in problem for problem in problems_for(payload))


@pytest.mark.parametrize("bad_id", ["C1", "1c", "with space", "", "c-1"])
def test_check_ids_must_be_slugs(bad_id: str) -> None:
    payload = valid_payload(
        checks=[{"id": bad_id, "question": "q", "method": "m"}]
    )
    assert any(".id must match" in problem for problem in problems_for(payload))


def test_a_check_without_a_question_is_rejected() -> None:
    payload = valid_payload(checks=[{"id": "c1", "method": "m"}])
    assert any("question" in problem for problem in problems_for(payload))


def test_a_check_without_a_method_is_rejected() -> None:
    payload = valid_payload(checks=[{"id": "c1", "question": "q"}])
    assert any("method" in problem for problem in problems_for(payload))


def test_dimensions_must_be_a_list_of_strings() -> None:
    payload = valid_payload(
        checks=[{"id": "c1", "question": "q", "method": "m", "dimensions": "region"}]
    )
    assert any("dimensions must be a list" in p for p in problems_for(payload))


def test_assumptions_must_be_a_list_of_strings() -> None:
    payload = valid_payload(assumptions="completed only")
    assert any("assumptions must be a list" in p for p in problems_for(payload))


def test_checks_must_be_a_list() -> None:
    payload = valid_payload(checks={"id": "c1"})
    assert any("checks must be a list" in p for p in problems_for(payload))


def test_every_problem_is_reported_at_once() -> None:
    payload = valid_payload(
        metric_definition="",
        answerable="maybe",
        timeframe={"start": "nope", "end": "2026-10-01", "comparison": ""},
        checks=[{"id": "BAD", "question": "", "method": ""}],
        assumptions=[1],
    )
    assert len(problems_for(payload)) >= 6


# --------------------------------------------------------------------------
# Unanswerable questions
# --------------------------------------------------------------------------


def test_an_unanswerable_plan_needs_a_gap_and_no_checks(
    settings: Settings, scripted_llm
) -> None:
    payload = valid_payload(
        answerable=False,
        checks=[],
        blocking_gap="there is no marketing spend table, so spend cannot be compared",
    )
    result, _client = plan(settings, scripted_llm, json.dumps(payload))

    assert result.status == "ok"
    restored = from_dict(InvestigationPlan, result.data)
    assert restored.answerable is False
    assert "marketing" in (restored.blocking_gap or "")
    assert restored.checks == ()
    assert "answerable: no" in result.text
    assert "blocking gap:" in result.text


def test_an_unanswerable_plan_without_a_gap_is_rejected() -> None:
    payload = valid_payload(answerable=False, checks=[], blocking_gap=None)
    assert any("blocking_gap" in problem for problem in problems_for(payload))


def test_an_unanswerable_plan_may_not_list_checks() -> None:
    payload = valid_payload(answerable=False, blocking_gap="no data")
    assert any("checks must be empty" in p for p in problems_for(payload))


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def test_the_summary_shows_methods_and_dimensions() -> None:
    text = render_plan_summary(parse_plan_payload(valid_payload(), max_checks=8))
    assert "timeframe: 2026-10-01 .. 2026-10-31" in text
    assert "granularity: month" in text
    assert "c1" in text and "c2" in text
    assert "method: sum line_total by month" in text
    assert "dimensions: month" in text
    assert "dimensions: region" in text


def test_the_summary_says_none_when_there_are_no_checks() -> None:
    payload = valid_payload(answerable=False, checks=[], blocking_gap="no data")
    text = render_plan_summary(parse_plan_payload(payload, max_checks=8))
    assert "checks: none" in text
