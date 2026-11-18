"""CLI behaviour: exit codes, dry-run output, config failure reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.main import (
    EXIT_CONFIG,
    EXIT_NOT_IMPLEMENTED,
    EXIT_OK,
    EXIT_RUN_FAILED,
    EXIT_SQL_FAILED,
    EXIT_SQL_REJECTED,
    EXIT_USAGE,
    main,
)
from agents.investigator import make_investigator
from agents.planner import make_planner
from agents.reporter import make_reporter
from llm.base import LLMError, LLMResponse, ToolCall
from tools.registry import build_toolset

QUESTION = "Why did our revenue drop this month?"

PLAN_PAYLOAD = {
    "metric_definition": "revenue = SUM(order_items.line_total) for completed orders",
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
            "question": "How did revenue change month over month?",
            "method": "sum line_total by month",
            "dimensions": ["month"],
        }
    ],
    "blocking_gap": None,
    "assumptions": ["completed orders only"],
}


_MOCK_COMPLETED_ORDERS = 2797

ANSWER_PAYLOAD = {
    "headline": "Completed orders rose.",
    "answer": f"The count of completed orders is {_MOCK_COMPLETED_ORDERS}.",
    "findings": [
        {
            "claim": "Completed orders are counted in the query.",
            "evidence_query_ids": ["q_001"],
            "numbers": {"orders": _MOCK_COMPLETED_ORDERS},
        }
    ],
    "assumptions": ["completed orders only"],
    "limitations": ["the mock data is synthetic"],
    "confidence": "medium",
}


ANSWER_PAYLOAD_6273 = ANSWER_PAYLOAD


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's .env must not influence these tests (G-02)."""
    monkeypatch.setattr("app.main.load_dotenv", lambda: None)


def test_dry_run_prints_a_validated_plan(configured: None, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--dry-run", QUESTION]) == EXIT_OK
    out = capsys.readouterr().out
    assert "plan: 3 steps" in out
    assert "validation: OK" in out
    assert "planner" in out and "investigator" in out and "reporter" in out


def test_dry_run_shows_effective_limits(configured: None, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--dry-run", QUESTION]) == EXIT_OK
    out = capsys.readouterr().out
    assert "rows 500" in out
    assert "backend: mock" in out


def test_dry_run_json_is_machine_readable(
    configured: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--dry-run", "--json", QUESTION]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["plan"]["steps"]) == 3
    assert [s["agent"] for s in payload["plan"]["steps"]] == [
        "planner",
        "investigator",
        "reporter",
    ]
    assert payload["question"] == QUESTION
    assert payload["settings"]["database_backend"] == "mock"


def test_dry_run_does_not_require_llm_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--dry-run", QUESTION]) == EXIT_OK
    assert "validation: OK" in capsys.readouterr().out


def test_missing_question_is_a_usage_error(configured: None) -> None:
    assert main(["--dry-run"]) == EXIT_USAGE


def test_config_errors_are_reported_with_their_variables(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    for name, value in {**env, "MAX_ROWS": "as-many-as-possible"}.items():
        monkeypatch.setenv(name, value)
    assert main(["--dry-run", QUESTION]) == EXIT_CONFIG
    err = capsys.readouterr().err
    assert "MAX_ROWS" in err
    assert ".env.example" in err


def test_a_real_run_requires_llm_credentials(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([QUESTION]) == EXIT_CONFIG
    err = capsys.readouterr().err
    assert "LLM_API_KEY" in err or "LLM_MODEL" in err


def test_a_question_without_credentials_is_a_configuration_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([QUESTION]) == EXIT_CONFIG
    assert "LLM_API_KEY is required" in capsys.readouterr().err


def test_schema_digest_is_printed(configured: None, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--show-schema"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "DATABASE SCHEMA" in out
    assert "orders" in out
    assert "-> orders.id" in out


def test_schema_digest_does_not_require_llm_credentials(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--show-schema"]) == EXIT_OK
    assert "DATABASE SCHEMA" in capsys.readouterr().out


def test_inspect_table_prints_detail(configured: None, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--inspect-table", "orders"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "orders" in out
    assert "value domains" in out
    assert "sample rows" in out


def test_inspect_table_rejects_unknown_names(
    configured: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--inspect-table", "revenue_by_month"]) == EXIT_USAGE
    err = capsys.readouterr().err
    assert "unknown table" in err
    assert "orders" in err


def test_inspect_table_rejects_sql_smuggling(
    configured: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--inspect-table", "orders; DROP TABLE orders"]) == EXIT_USAGE
    assert "unknown table" in capsys.readouterr().err


def test_query_runs_against_the_mock_warehouse(
    configured: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--query", "SELECT COUNT(*) AS n FROM orders"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "status: ok" in out
    assert "columns: n" in out
    assert "row_count: 1" in out


def test_query_does_not_require_llm_credentials(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--query", "SELECT 1 AS one"]) == EXIT_OK
    assert "status: ok" in capsys.readouterr().out


def test_query_refuses_writes(configured: None, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--query", "DROP TABLE orders"]) == EXIT_SQL_REJECTED
    err = capsys.readouterr().err
    assert "only read-only" in err
    assert "rule: not_select" in err


def test_query_reports_a_database_error_with_an_id(
    configured: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--query", "SELECT nope FROM orders"]) == EXIT_SQL_FAILED
    err = capsys.readouterr().err
    assert "no such column" in err.lower()
    assert "error_id" in err


def test_query_explain_prints_the_full_failure(
    configured: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--query", "SELECT nope FROM orders", "--explain"]) == EXIT_SQL_FAILED
    err = capsys.readouterr().err
    assert "columns of orders:" in err
    assert "customer_id" in err


def test_query_reports_truncation(configured: None, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--query", "SELECT id FROM order_items"]) == EXIT_OK
    assert "truncated: yes" in capsys.readouterr().out


def test_explain_requires_a_query(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--explain", QUESTION]) == EXIT_USAGE
    assert "--explain only applies to --query" in capsys.readouterr().err


# --------------------------------------------------------------------------
# --plan: one real model call, through the orchestrator
# --------------------------------------------------------------------------


def scripted_planner(monkeypatch: pytest.MonkeyPatch, scripted_llm, *replies: object):
    """Wire a scripted model in place of the provider, keeping everything else real."""
    client = scripted_llm(list(replies))
    monkeypatch.setattr(
        "wire.build_agents",
        lambda settings, database: {"planner": make_planner(client)},
    )
    return client


def test_plan_requires_llm_credentials(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--plan", QUESTION]) == EXIT_CONFIG
    assert "LLM_API_KEY is required" in capsys.readouterr().err


def test_plan_prints_the_investigation_plan(
    configured: None,
    monkeypatch: pytest.MonkeyPatch,
    scripted_llm,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scripted_planner(monkeypatch, scripted_llm, json.dumps(PLAN_PAYLOAD))

    assert main(["--plan", QUESTION]) == EXIT_OK
    out = capsys.readouterr().out
    assert "metric: revenue = SUM" in out
    assert "answerable: yes" in out
    assert "checks: 1" in out
    assert "task:" in out
    assert "tokens: 10 in / 5 out" in out


def test_plan_json_is_machine_readable(
    configured: None,
    monkeypatch: pytest.MonkeyPatch,
    scripted_llm,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scripted_planner(monkeypatch, scripted_llm, json.dumps(PLAN_PAYLOAD))

    assert main(["--plan", "--json", QUESTION]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["question"] == QUESTION
    assert payload["plan"]["timeframe"]["granularity"] == "month"
    assert payload["plan"]["checks"][0]["id"] == "c1"
    assert payload["tokens_in"] == 10
    assert payload["settings"]["llm_model"] == "test-model"


def test_plan_writes_one_trace_event(
    configured: None,
    monkeypatch: pytest.MonkeyPatch,
    scripted_llm,
    trace_dir: Path,
) -> None:
    scripted_planner(monkeypatch, scripted_llm, json.dumps(PLAN_PAYLOAD))

    assert main(["--plan", QUESTION]) == EXIT_OK
    files = list(trace_dir.glob("*.jsonl"))
    assert len(files) == 1
    events = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]
    assert len(events) == 1
    assert events[0]["step_id"] == "plan"
    assert events[0]["agent"] == "planner"
    assert events[0]["outcome"] == "ok"
    assert events[0]["tokens_in"] == 10
    assert events[0]["model"] == "test-model"
    assert len(events[0]["prompt_hash"]) == 16


def test_the_prompt_hash_identifies_the_prompt_without_storing_it(
    configured: None,
    monkeypatch: pytest.MonkeyPatch,
    scripted_llm,
    trace_dir: Path,
) -> None:
    def run_and_hash(question: str) -> str:
        scripted_planner(monkeypatch, scripted_llm, json.dumps(PLAN_PAYLOAD))
        assert main(["--plan", question]) == EXIT_OK
        files = list(trace_dir.glob("*.jsonl"))
        assert len(files) == 1
        event = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
        files[0].unlink()
        return event["prompt_hash"]

    first = run_and_hash(QUESTION)
    again = run_and_hash(QUESTION)
    other = run_and_hash("How many orders did we take in October 2026?")

    assert first == again
    assert first != other


def test_plan_reports_a_model_failure(
    configured: None,
    monkeypatch: pytest.MonkeyPatch,
    scripted_llm,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scripted_planner(
        monkeypatch, scripted_llm, LLMError("auth", "invalid api key", retryable=False)
    )

    assert main(["--plan", QUESTION]) == EXIT_RUN_FAILED
    assert "llm_error: auth: invalid api key" in capsys.readouterr().err


def test_plan_reports_an_unrepairable_plan(
    configured: None,
    monkeypatch: pytest.MonkeyPatch,
    scripted_llm,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scripted_planner(monkeypatch, scripted_llm, "no json", "still no json")

    assert main(["--plan", QUESTION]) == EXIT_RUN_FAILED
    assert "plan_invalid" in capsys.readouterr().err


def test_plan_sends_the_question_and_the_schema_digest(
    configured: None,
    monkeypatch: pytest.MonkeyPatch,
    scripted_llm,
) -> None:
    client = scripted_planner(monkeypatch, scripted_llm, json.dumps(PLAN_PAYLOAD))

    assert main(["--plan", QUESTION]) == EXIT_OK
    context = client.requests[0].messages[0].content
    assert QUESTION in context
    assert "DATABASE SCHEMA" in context
    assert "orders" in context
    assert "as of" in context


def test_dry_run_and_plan_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--dry-run", "--plan", QUESTION])
    assert caught.value.code == 2


def test_trace_dir_override_is_applied(
    configured: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    override = tmp_path / "elsewhere"
    assert main(["--dry-run", "--json", "--trace-dir", str(override), QUESTION]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["trace_dir"] == str(override)


def test_dry_run_writes_no_trace_file(configured: None, trace_dir: Path) -> None:
    assert main(["--dry-run", QUESTION]) == EXIT_OK
    assert not trace_dir.exists() or list(trace_dir.iterdir()) == []


# --------------------------------------------------------------------------
# A question: plan, then investigate, through the real tools
# --------------------------------------------------------------------------

COUNT_SQL = "SELECT COUNT(*) AS completed_orders FROM orders WHERE status = 'completed'"


def scripted_run(
    monkeypatch: pytest.MonkeyPatch,
    scripted_llm,
    *replies: object,
    reporter_replies: tuple[str, ...] = (),
):
    """Both agents on one scripted client, but the tools run for real.

    The toolset is the real one, bound to the real in-memory warehouse, so these
    tests exercise the guard, the executor, and the evidence records end to end.
    """
    client = scripted_llm(list(replies) + list(reporter_replies))

    def build(settings, database):
        return {
            "planner": make_planner(client),
            "investigator": make_investigator(
                client, build_toolset(database, settings)
            ),
            "reporter": make_reporter(client),
        }

    monkeypatch.setattr("wire.build_agents", build)
    return client


def call_execute(sql: str, *, call_id: str = "call_1") -> LLMResponse:
    return LLMResponse(
        text="",
        tool_calls=(
            ToolCall(id=call_id, name="execute_sql", arguments={"sql": sql}),
        ),
        finish_reason="tool_calls",
    )


# The mock warehouse (seed 7) returns 5982 completed orders. Every scripted finding
# must cite a number that actually appears in that row, or the investigator's
# number validator rejects it (Phase 8).
_MOCK_COMPLETED_ORDERS = 2797


def findings_reply(
    *, evidence: str = "q_001", numbers: dict | None = None
) -> str:
    return json.dumps(
        {
            "findings": [
                {
                    "id": "f1",
                    "claim": "Completed orders are counted in the query.",
                    "evidence_query_ids": [evidence],
                    "numbers": (
                        numbers
                        if numbers is not None
                        else {"orders": _MOCK_COMPLETED_ORDERS}
                    ),
                }
            ],
            "open_questions": [],
            "notes": [],
        }
    )


def test_a_question_runs_plan_and_investigation(
    configured: None,
    monkeypatch: pytest.MonkeyPatch,
    scripted_llm,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scripted_run(
        monkeypatch,
        scripted_llm,
        json.dumps(PLAN_PAYLOAD),
        call_execute(COUNT_SQL),
        findings_reply(),
        reporter_replies=(json.dumps(ANSWER_PAYLOAD),),
    )

    assert main([QUESTION]) == EXIT_OK
    out = capsys.readouterr().out
    assert "plan" in out
    assert "metric: revenue = SUM" in out
    assert "investigation" in out
    assert "findings: 1" in out
    assert "q_001" in out
    assert COUNT_SQL in out
    assert "stop reason: model_finished" in out
    assert "answer" in out
    assert "Completed orders rose." in out
    assert "confidence: medium" in out
    assert "limitations:" in out


def test_a_question_json_carries_the_evidence(
    configured: None,
    monkeypatch: pytest.MonkeyPatch,
    scripted_llm,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scripted_run(
        monkeypatch,
        scripted_llm,
        json.dumps(PLAN_PAYLOAD),
        call_execute(COUNT_SQL),
        findings_reply(),
        reporter_replies=(json.dumps(ANSWER_PAYLOAD),),
    )

    assert main(["--json", QUESTION]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["metric_definition"].startswith("revenue")
    assert payload["investigation"]["findings"][0]["numbers"] == {"orders": 2797.0}
    queries = payload["investigation"]["queries"]
    assert len(queries) == 1
    assert queries[0]["status"] == "ok"
    assert queries[0]["id"] == "q_001"
    assert queries[0]["row_count"] == 1
    assert "completed_orders" in queries[0]["preview"]
    assert payload["tokens_in"] > 0
    assert payload["answer"]["headline"] == "Completed orders rose."


def test_a_question_writes_one_trace_event_per_step(
    configured: None,
    monkeypatch: pytest.MonkeyPatch,
    scripted_llm,
    trace_dir: Path,
) -> None:
    scripted_run(
        monkeypatch,
        scripted_llm,
        json.dumps(PLAN_PAYLOAD),
        call_execute(COUNT_SQL),
        findings_reply(),
        reporter_replies=(json.dumps(ANSWER_PAYLOAD),),
    )

    assert main([QUESTION]) == EXIT_OK
    files = list(trace_dir.glob("*.jsonl"))
    assert len(files) == 1
    events = [
        json.loads(line)
        for line in files[0].read_text(encoding="utf-8").splitlines()
    ]
    assert [event["step_id"] for event in events] == ["plan", "investigate", "answer"]
    assert [event["agent"] for event in events] == ["planner", "investigator", "reporter"]
    assert events[1]["tool_calls"] == ["execute_sql"]
    assert all(event["outcome"] == "ok" for event in events)


def test_a_failed_investigation_still_shows_the_sql_it_ran(
    configured: None,
    monkeypatch: pytest.MonkeyPatch,
    scripted_llm,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scripted_run(
        monkeypatch,
        scripted_llm,
        json.dumps(PLAN_PAYLOAD),
        call_execute(COUNT_SQL),
        "no json here",
        "still no json here",
    )

    assert main([QUESTION]) == EXIT_RUN_FAILED
    err = capsys.readouterr().err
    assert "report_invalid" in err
    assert "queries attempted: 1" in err
    assert COUNT_SQL in err


def test_an_unrepairable_finding_is_not_silently_accepted(
    configured: None,
    monkeypatch: pytest.MonkeyPatch,
    scripted_llm,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A claim citing a query that never ran is rejected, twice, then fails."""
    scripted_run(
        monkeypatch,
        scripted_llm,
        json.dumps(PLAN_PAYLOAD),
        call_execute(COUNT_SQL),
        findings_reply(evidence="q_999"),
        findings_reply(evidence="q_999"),
    )

    assert main([QUESTION]) == EXIT_RUN_FAILED
    assert "q_999" in capsys.readouterr().err
