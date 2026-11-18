"""The investigator: the loop, the budgets, and the findings contract.

The tools are real (they run against the in-memory warehouse); the model is a
`ScriptedLLM`. That split is the point of this file: what the model does is scripted,
what the code guarantees is not.
"""

from __future__ import annotations

import json

import pytest

from agents.investigator import (
    FindingsInvalid,
    make_investigator,
    parse_findings_payload,
)
from core.config import Settings
from core.types import InvestigationReport, from_dict
from db.base import Database
from db.factory import build_database
from llm.base import LLMError, LLMResponse, Message, ToolCall
from tools.registry import build_toolset

CONTEXT = (
    "[step: investigate]\nProduce evidence for the question.\n\n"
    "[artifact: plan]\nmetric: revenue\nanswerable: yes"
)


@pytest.fixture
def database(settings: Settings) -> Database:
    return build_database(settings)


def execute_call(
    sql: str = "SELECT COUNT(*) AS n FROM orders", *, call_id: str = "call_1"
) -> ToolCall:
    return ToolCall(id=call_id, name="execute_sql", arguments={"sql": sql})


def tool_turn(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(
        text="", tool_calls=calls, finish_reason="tool_calls", tokens_in=10, tokens_out=5
    )


def findings_payload(
    *,
    evidence: tuple[str, ...] = ("q_001",),
    numbers: dict | None = None,
    **overrides: object,
) -> str:
    payload: dict[str, object] = {
        "findings": [
            {
                "id": "f1",
                "claim": "Completed orders were counted.",
                "evidence_query_ids": list(evidence),
                "numbers": {"orders": 2938} if numbers is None else numbers,
            }
        ],
        "open_questions": [],
        "notes": [],
    }
    payload.update(overrides)
    return json.dumps(payload)


def run(settings: Settings, database: Database, scripted_llm, *replies: object):
    client = scripted_llm(list(replies))
    toolset = build_toolset(database, settings)
    result = make_investigator(client, toolset)(CONTEXT, settings)
    return result, client, toolset


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def test_a_tool_turn_then_findings_becomes_a_report(
    settings: Settings, database: Database, scripted_llm
) -> None:
    result, client, toolset = run(
        settings,
        database,
        scripted_llm,
        tool_turn(execute_call()),
        findings_payload(numbers={"orders": 2938}),
    )

    assert result.status == "ok"
    assert result.error is None
    assert result.tool_calls == ("execute_sql",)
    assert (result.tokens_in, result.tokens_out) == (20, 10)

    report = from_dict(InvestigationReport, result.data or {})
    assert report.stop_reason == "model_finished"
    assert [finding.id for finding in report.findings] == ["f1"]
    assert report.findings[0].numbers == {"orders": 2938.0}
    assert report.findings[0].evidence_query_ids == ("q_001",)
    assert [record.status for record in report.queries] == ["ok"]
    assert report.queries[0].row_count == 1
    assert len(client.requests) == 2


def test_the_first_request_carries_the_context_and_the_tools(
    settings: Settings, database: Database, scripted_llm
) -> None:
    _result, client, toolset = run(
        settings, database, scripted_llm, tool_turn(execute_call()), findings_payload()
    )

    request = client.requests[0]
    assert request.messages == (Message(role="user", content=CONTEXT),)
    assert request.tools == toolset.specs
    assert request.json_mode is False
    assert request.max_tokens == settings.llm_max_tokens


def test_the_tool_result_is_sent_back_as_a_tool_message(
    settings: Settings, database: Database, scripted_llm
) -> None:
    _result, client, _toolset = run(
        settings,
        database,
        scripted_llm,
        tool_turn(execute_call()),
        findings_payload(),
    )

    messages = client.requests[1].messages
    assert [message.role for message in messages] == ["user", "assistant", "tool"]
    assert messages[1].tool_calls[0].name == "execute_sql"
    assert messages[2].tool_call_id == "call_1"
    assert messages[2].name == "execute_sql"
    assert "query_id: q_001" in messages[2].content
    assert "status: ok" in messages[2].content


def test_the_instruction_states_the_real_limits(
    settings: Settings, database: Database, scripted_llm
) -> None:
    _result, client, _toolset = run(
        settings, database, scripted_llm, tool_turn(execute_call()), findings_payload()
    )
    system = client.requests[0].system
    assert f"at most {settings.max_queries} queries" in system
    assert f"{settings.max_tool_turns} turns" in system
    assert f"{settings.max_rows} rows" in system
    assert f"{settings.statement_timeout_ms} ms" in system


def test_the_instruction_states_the_findings_contract(
    settings: Settings, database: Database, scripted_llm
) -> None:
    _result, client, _toolset = run(
        settings, database, scripted_llm, tool_turn(execute_call()), findings_payload()
    )
    system = client.requests[0].system
    assert "evidence_query_ids" in system
    assert "a claim without evidence is rejected" in system
    assert "A change is not a cause" in system
    assert "cannot be answered with this schema" in system


def test_a_model_that_finishes_on_its_own_needs_no_extra_call(
    settings: Settings, database: Database, scripted_llm
) -> None:
    """A model that never calls a tool is answered once, not twice."""
    no_evidence_possible = json.dumps(
        {
            "findings": [],
            "open_questions": ["the schema has no marketing spend table"],
            "notes": [],
        }
    )
    result, client, _toolset = run(
        settings, database, scripted_llm, no_evidence_possible
    )
    assert result.status == "ok"
    assert len(client.requests) == 1
    report = from_dict(InvestigationReport, result.data or {})
    assert report.findings == ()
    assert report.open_questions == ("the schema has no marketing spend table",)


def test_tool_names_are_recorded_in_order(
    settings: Settings, database: Database, scripted_llm
) -> None:
    result, _client, _toolset = run(
        settings,
        database,
        scripted_llm,
        tool_turn(execute_call()),
        tool_turn(ToolCall(id="call_2", name="inspect_schema", arguments={"table": "orders"})),
        findings_payload(),
    )
    assert result.tool_calls == ("execute_sql", "inspect_schema")


def test_tokens_are_summed_over_turns(
    settings: Settings, database: Database, scripted_llm
) -> None:
    result, _client, _toolset = run(
        settings,
        database,
        scripted_llm,
        tool_turn(execute_call()),
        tool_turn(execute_call("SELECT 2 AS two", call_id="call_2")),
        findings_payload(evidence=("q_001", "q_002")),
    )
    assert (result.tokens_in, result.tokens_out) == (30, 15)


# --------------------------------------------------------------------------
# Budgets
# --------------------------------------------------------------------------


def test_the_turn_budget_forces_a_final_answer(
    env: dict[str, str], database: Database, scripted_llm
) -> None:
    tight = Settings.from_env({**env, "MAX_TOOL_TURNS": "1"})
    result, client, _toolset = run(
        tight,
        database,
        scripted_llm,
        tool_turn(execute_call()),
        findings_payload(),
    )

    assert result.status == "ok"
    final = client.requests[1]
    assert final.tools == ()
    assert "budget is exhausted" in final.messages[-1].content

    report = from_dict(InvestigationReport, result.data or {})
    assert report.stop_reason == "turn_budget"
    assert any("turn budget" in note for note in report.notes)


def test_the_query_budget_refuses_the_rest_and_still_reports(
    env: dict[str, str], database: Database, scripted_llm
) -> None:
    tight = Settings.from_env({**env, "MAX_QUERIES": "1"})
    result, client, _toolset = run(
        tight,
        database,
        scripted_llm,
        tool_turn(
            execute_call("SELECT 1 AS one", call_id="call_1"),
            execute_call("SELECT 2 AS two", call_id="call_2"),
        ),
        findings_payload(numbers={"orders": 1}),
    )

    assert result.status == "ok"
    assert client.requests[1].tools == ()
    report = from_dict(InvestigationReport, result.data or {})
    assert report.stop_reason == "query_budget"
    assert [record.id for record in report.queries] == ["q_001", "x1"]
    assert report.queries[1].error_kind == "not_run"
    assert any("query budget" in note for note in report.notes)
    assert any("refused" in note for note in report.notes)


def test_a_failed_query_is_recorded_but_cannot_be_cited(
    settings: Settings, database: Database, scripted_llm
) -> None:
    """The failure is evidence of a mistake, not evidence for a claim."""
    result, client, _toolset = run(
        settings,
        database,
        scripted_llm,
        tool_turn(execute_call("SELECT nope FROM orders")),
        findings_payload(evidence=("err_001",), numbers={}),
        findings_payload(evidence=("err_001",), numbers={}),
    )

    assert result.status == "error"
    assert result.error is not None
    assert "err_001" in result.error
    assert "not a query that returned results" in result.error
    assert "err_001" in client.requests[1].messages[-1].content

    report = from_dict(InvestigationReport, result.data or {})
    assert report.queries[0].id == "err_001"
    assert report.queries[0].error_kind == "invalid_query"


# --------------------------------------------------------------------------
# Findings contract
# --------------------------------------------------------------------------


def test_findings_citing_an_unknown_query_are_sent_back(
    settings: Settings, database: Database, scripted_llm
) -> None:
    result, client, _toolset = run(
        settings,
        database,
        scripted_llm,
        tool_turn(execute_call()),
        findings_payload(evidence=("q_999",)),
        findings_payload(evidence=("q_001",)),
    )

    assert result.status == "ok"
    assert len(client.requests) == 3
    repair = client.requests[2]
    assert repair.json_mode is True
    assert repair.tools == ()
    assert [message.role for message in repair.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert "q_999" in repair.messages[-1].content


def test_a_claim_without_evidence_is_sent_back(
    settings: Settings, database: Database, scripted_llm
) -> None:
    result, client, _toolset = run(
        settings,
        database,
        scripted_llm,
        tool_turn(execute_call()),
        findings_payload(evidence=()),
        findings_payload(),
    )
    assert result.status == "ok"
    assert "at least one query_id" in client.requests[2].messages[-1].content


def test_numbers_that_are_not_numbers_are_sent_back(
    settings: Settings, database: Database, scripted_llm
) -> None:
    result, client, _toolset = run(
        settings,
        database,
        scripted_llm,
        tool_turn(execute_call()),
        findings_payload(numbers={"orders": "about six thousand"}),
        findings_payload(),
    )
    assert result.status == "ok"
    assert "must be a number" in client.requests[2].messages[-1].content


def test_a_blank_reply_is_repaired_without_an_empty_turn(
    settings: Settings, database: Database, scripted_llm
) -> None:
    """A model with nothing to say is asked again, and no empty turn is stored."""
    nothing_to_cite = json.dumps(
        {"findings": [], "open_questions": ["no query has run yet"], "notes": []}
    )
    result, client, _toolset = run(
        settings, database, scripted_llm, "   ", nothing_to_cite
    )
    assert result.status == "ok"
    assert [message.role for message in client.requests[1].messages] == ["user", "user"]


def test_a_second_failure_keeps_the_queries_it_paid_for(
    settings: Settings, database: Database, scripted_llm
) -> None:
    result, client, _toolset = run(
        settings,
        database,
        scripted_llm,
        tool_turn(execute_call()),
        "no json",
        "still no json",
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.startswith("report_invalid:")
    assert result.tool_calls == ("execute_sql",)  # the trace keeps what was tried
    assert len(client.requests) == 3
    report = from_dict(InvestigationReport, result.data or {})
    assert [record.id for record in report.queries] == ["q_001"]
    assert report.stop_reason == "report_invalid"


def test_an_empty_report_is_rejected(
    settings: Settings, database: Database, scripted_llm
) -> None:
    empty = json.dumps({"findings": [], "open_questions": [], "notes": []})
    result, client, _toolset = run(
        settings, database, scripted_llm, empty, empty
    )
    assert result.status == "error"
    assert "says nothing" in client.requests[1].messages[-1].content


def test_open_questions_and_notes_are_kept(
    settings: Settings, database: Database, scripted_llm
) -> None:
    result, _client, _toolset = run(
        settings,
        database,
        scripted_llm,
        tool_turn(execute_call()),
        findings_payload(
            open_questions=["inventory data would test the stockout hypothesis"],
            notes=["October has 31 days; September 30"],
        ),
    )
    report = from_dict(InvestigationReport, result.data or {})
    assert report.open_questions == ("inventory data would test the stockout hypothesis",)
    assert report.notes[0] == "October has 31 days; September 30"
    assert "open questions:" in result.text
    assert "notes:" in result.text


def test_the_artifact_lists_evidence_and_the_query_log(
    settings: Settings, database: Database, scripted_llm
) -> None:
    result, _client, _toolset = run(
        settings,
        database,
        scripted_llm,
        tool_turn(execute_call()),
        findings_payload(),
    )
    assert "findings: 1" in result.text
    assert "f1  [q_001]" in result.text
    assert "numbers: orders=2938" in result.text
    assert f"queries: 1 attempted of {settings.max_queries} allowed" in result.text
    assert "SELECT COUNT(*) AS n FROM orders" in result.text
    assert "stop reason: model_finished" in result.text


# --------------------------------------------------------------------------
# Model failures
# --------------------------------------------------------------------------


def test_a_model_failure_inside_the_loop_is_an_agent_error(
    settings: Settings, database: Database, scripted_llm
) -> None:
    result, client, _toolset = run(
        settings, database, scripted_llm, LLMError("rate_limit", "slow down")
    )
    assert result.status == "error"
    assert result.error == "llm_error: rate_limit: slow down"
    assert len(client.requests) == 1
    report = from_dict(InvestigationReport, result.data or {})
    assert report.queries == ()
    assert report.stop_reason == "llm_error"


def test_a_model_failure_on_the_final_ask_is_an_agent_error(
    env: dict[str, str], database: Database, scripted_llm
) -> None:
    tight = Settings.from_env({**env, "MAX_TOOL_TURNS": "1"})
    result, _client, _toolset = run(
        tight,
        database,
        scripted_llm,
        tool_turn(execute_call()),
        LLMError("timeout", "no answer"),
    )
    assert result.status == "error"
    assert result.error == "llm_error: timeout: no answer"
    report = from_dict(InvestigationReport, result.data or {})
    assert [record.id for record in report.queries] == ["q_001"]


# --------------------------------------------------------------------------
# The payload parser, in isolation
# --------------------------------------------------------------------------


def payload(**overrides: object) -> dict:
    base: dict[str, object] = {
        "findings": [
            {
                "id": "f1",
                "claim": "Revenue fell.",
                "evidence_query_ids": ["q_001"],
                "numbers": {"change_pct": -11.8},
            }
        ],
        "open_questions": [],
        "notes": [],
    }
    base.update(overrides)
    return base


def problems_for(value: object, *, evidence: tuple[str, ...] = ("q_001",)) -> list[str]:
    with pytest.raises(FindingsInvalid) as caught:
        parse_findings_payload(value, evidence=evidence)
    return list(caught.value.problems)


def test_a_valid_payload_is_accepted() -> None:
    findings, questions, notes = parse_findings_payload(payload(), evidence=("q_001",))
    assert findings[0].id == "f1"
    assert findings[0].numbers == {"change_pct": -11.8}
    assert questions == () and notes == ()


def test_a_failed_query_cannot_be_evidence() -> None:
    problems = problems_for(payload(), evidence=())
    assert any("not a query that returned results" in problem for problem in problems)


def test_duplicate_finding_ids_are_rejected() -> None:
    findings = [
        {"id": "f1", "claim": "a", "evidence_query_ids": ["q_001"]},
        {"id": "f1", "claim": "b", "evidence_query_ids": ["q_001"]},
    ]
    assert any("duplicated" in problem for problem in problems_for(payload(findings=findings)))


def test_finding_ids_must_be_slugs() -> None:
    findings = [{"id": "F1", "claim": "a", "evidence_query_ids": ["q_001"]}]
    assert any(".id must match" in problem for problem in problems_for(payload(findings=findings)))


def test_too_many_findings_are_rejected() -> None:
    findings = [
        {"id": f"f{index}", "claim": "a", "evidence_query_ids": ["q_001"]}
        for index in range(1, 12)
    ]
    assert any("more than 10" in problem for problem in problems_for(payload(findings=findings)))


def test_booleans_are_not_numbers() -> None:
    findings = [
        {
            "id": "f1",
            "claim": "a",
            "evidence_query_ids": ["q_001"],
            "numbers": {"flag": True},
        }
    ]
    assert any("must be a number" in problem for problem in problems_for(payload(findings=findings)))


def test_an_empty_report_is_rejected_by_the_parser() -> None:
    problems = problems_for({"findings": [], "open_questions": [], "notes": []})
    assert any("says nothing" in problem for problem in problems)


def test_a_non_object_payload_is_rejected() -> None:
    assert problems_for(["f1"]) == ["the reply is not a JSON object"]
