"""Tool dispatch: specs, argument handling, budgets, and the evidence records.

The database is the real mock warehouse, so guard refusals and driver errors are
the real ones; only the model is absent (this layer never sees a model).
"""

from __future__ import annotations

import pytest

from core.config import Settings
from db.base import Database
from db.factory import build_database
from llm.base import ToolCall
from tools.registry import ToolOutcome, Toolset, build_toolset


@pytest.fixture
def database(settings: Settings) -> Database:
    return build_database(settings)


@pytest.fixture
def toolset(database: Database, settings: Settings) -> Toolset:
    return build_toolset(database, settings)


def call(name: str, arguments: dict, *, call_id: str = "call_1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def count(toolset: Toolset) -> ToolOutcome:
    return toolset.call(call("execute_sql", {"sql": "SELECT COUNT(*) AS n FROM orders"}))


# --------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------


def test_three_tools_are_offered(toolset: Toolset) -> None:
    assert [spec.name for spec in toolset.specs] == [
        "execute_sql",
        "inspect_schema",
        "get_query_error_details",
    ]


def test_the_sql_tool_states_the_real_limits(toolset: Toolset, settings: Settings) -> None:
    spec = toolset.specs[0]
    assert str(settings.max_rows) in spec.description
    assert str(settings.statement_timeout_ms) in spec.description
    assert spec.parameters["required"] == ["sql"]


def test_every_spec_declares_its_required_arguments(toolset: Toolset) -> None:
    required = {spec.name: spec.parameters["required"] for spec in toolset.specs}
    assert required["inspect_schema"] == ["table"]
    assert required["get_query_error_details"] == ["error_id"]


# --------------------------------------------------------------------------
# A statement that runs
# --------------------------------------------------------------------------


def test_a_successful_query_becomes_a_citable_record(toolset: Toolset) -> None:
    outcome = count(toolset)

    assert outcome.ok is True
    assert "status: ok" in outcome.text
    record = outcome.record
    assert record is not None
    assert record.status == "ok"
    assert record.row_count == 1
    assert record.columns == ("n",)
    assert record.id == "q_001"
    assert f"query_id: {record.id}" in outcome.text
    assert toolset.queries_used == 1
    assert toolset.records == [record]


def test_the_recorded_sql_is_the_statement_that_ran(toolset: Toolset) -> None:
    outcome = toolset.call(call("execute_sql", {"sql": "SELECT 1 AS one;"}))
    assert outcome.record is not None
    assert outcome.record.sql == "SELECT 1 AS one"


def test_a_result_preview_is_bounded(toolset: Toolset, settings: Settings) -> None:
    outcome = toolset.call(call("execute_sql", {"sql": "SELECT * FROM order_items LIMIT 40"}))
    assert outcome.record is not None
    preview = outcome.record.preview
    lines = preview.splitlines()
    assert len(preview) <= 800
    assert len(lines) == settings.trace_sample_rows + 2  # header + rows + note
    assert " | " in lines[0]
    assert lines[-1] == f"({40 - settings.trace_sample_rows} more rows)"


def test_an_empty_result_is_a_success(toolset: Toolset) -> None:
    outcome = toolset.call(call("execute_sql", {"sql": "SELECT 1 AS one WHERE 1 = 0"}))
    assert outcome.ok is True
    assert outcome.record is not None
    assert outcome.record.row_count == 0
    assert outcome.record.preview == ""
    assert "no rows" in outcome.text


# --------------------------------------------------------------------------
# Refusals and failures
# --------------------------------------------------------------------------


def test_a_guard_refusal_never_reaches_the_database(toolset: Toolset) -> None:
    outcome = toolset.call(call("execute_sql", {"sql": "DELETE FROM orders"}))

    assert outcome.ok is False
    assert "kind: rejected" in outcome.text
    record = outcome.record
    assert record is not None
    assert record.status == "error"
    assert record.error_kind == "rejected"
    assert record.id == "r1"
    assert toolset.queries_used == 0  # charged to the failure allowance instead
    assert toolset.failures == 1


def test_a_driver_failure_is_recorded_with_its_error_id(toolset: Toolset) -> None:
    outcome = toolset.call(call("execute_sql", {"sql": "SELECT nope FROM orders"}))

    assert outcome.ok is False
    record = outcome.record
    assert record is not None
    assert record.status == "error"
    assert record.error_kind == "invalid_query"
    assert record.id == "err_001"
    assert "error_id" in outcome.text
    assert "get_query_error_details" in outcome.text


def test_the_query_budget_refuses_extra_statements(env: dict[str, str], database: Database) -> None:
    tight = Settings.from_env({**env, "MAX_QUERIES": "1"})
    toolset = build_toolset(database, tight)

    assert count(toolset).ok is True
    refused = toolset.call(call("execute_sql", {"sql": "SELECT 2 AS two"}))

    assert refused.ok is False
    assert "query budget exhausted" in refused.text
    record = refused.record
    assert record is not None
    assert record.error_kind == "not_run"
    assert record.id == "x1"
    assert record.sql == "SELECT 2 AS two"  # intent is recorded, execution is not
    assert toolset.queries_used == 1


def test_the_failure_allowance_is_one_attempt_plus_retries(
    env: dict[str, str], database: Database
) -> None:
    tight = Settings.from_env({**env, "MAX_QUERY_RETRIES": "0"})
    toolset = build_toolset(database, tight)
    assert toolset.failure_allowance == 1

    assert toolset.call(call("execute_sql", {"sql": "SELECT nope FROM orders"})).ok is False
    refused = toolset.call(call("execute_sql", {"sql": "SELECT 1 AS one"}))

    assert refused.ok is False
    assert "too many failed queries" in refused.text
    assert toolset.queries_used == 1  # the failed attempt still reached the database
    assert toolset.refused == 1


def test_budget_exhaustion_is_visible_to_the_caller(
    env: dict[str, str], database: Database
) -> None:
    tight = Settings.from_env({**env, "MAX_QUERIES": "1"})
    toolset = build_toolset(database, tight)
    assert toolset.budget_exhausted is False

    count(toolset)
    assert toolset.budget_exhausted is True


# --------------------------------------------------------------------------
# Calls that never reach the tool
# --------------------------------------------------------------------------


def test_an_unknown_tool_names_the_available_ones(toolset: Toolset) -> None:
    outcome = toolset.call(call("drop_everything", {}))
    assert outcome.ok is False
    assert outcome.record is None
    assert "no such tool" in outcome.text
    assert "execute_sql" in outcome.text


def test_unreadable_arguments_are_reported(toolset: Toolset) -> None:
    outcome = toolset.call(
        ToolCall(
            id="call_1",
            name="execute_sql",
            arguments={},
            raw_arguments="{oops",
            parse_error="arguments are not valid JSON: Expecting property name",
        )
    )
    assert outcome.ok is False
    assert "could not be read" in outcome.text
    assert toolset.queries_used == 0


def test_a_missing_sql_argument_is_explained(toolset: Toolset) -> None:
    outcome = toolset.call(call("execute_sql", {"query": "SELECT 1"}))
    assert outcome.ok is False
    assert '"sql"' in outcome.text
    assert toolset.records == []


# --------------------------------------------------------------------------
# inspect_schema and error forensics
# --------------------------------------------------------------------------


def test_inspect_schema_does_not_consume_the_query_budget(toolset: Toolset) -> None:
    outcome = toolset.call(call("inspect_schema", {"table": "orders"}))

    assert outcome.ok is True
    assert outcome.record is None
    assert "orders" in outcome.text
    assert toolset.queries_used == 0


def test_inspect_schema_rejects_an_unknown_table(toolset: Toolset) -> None:
    outcome = toolset.call(call("inspect_schema", {"table": "ghost"}))
    assert outcome.ok is False
    assert "ghost" in outcome.text


def test_error_details_explain_a_recorded_failure(toolset: Toolset) -> None:
    failure = toolset.call(call("execute_sql", {"sql": "SELECT nope FROM orders"}))
    assert failure.record is not None

    details = toolset.call(
        call("get_query_error_details", {"error_id": failure.record.id})
    )

    assert details.ok is True
    assert "no such column" in details.text
    assert toolset.queries_used == 1  # forensics are free


def test_error_details_reject_an_unknown_id(toolset: Toolset) -> None:
    outcome = toolset.call(call("get_query_error_details", {"error_id": "err_999"}))
    assert outcome.ok is False
    assert "err_999" in outcome.text


def test_records_keep_execution_order(toolset: Toolset) -> None:
    toolset.call(call("execute_sql", {"sql": "SELECT 1 AS a"}))
    toolset.call(call("execute_sql", {"sql": "SELECT nope FROM orders"}))
    toolset.call(call("execute_sql", {"sql": "SELECT 2 AS b"}))

    assert [record.id for record in toolset.records] == ["q_001", "err_001", "q_002"]
