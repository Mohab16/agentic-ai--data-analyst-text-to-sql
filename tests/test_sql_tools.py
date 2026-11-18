"""The SQL tools: execution, limits, failure forensics, and rendering."""

from __future__ import annotations

import inspect
import re
from dataclasses import replace
from datetime import date

import pytest

from core.config import Settings
from core.types import ToolError
from db.base import ErrorJournal, QueryError, QueryResult
from db.mock_source import MockDatabase
from tools.sql import (
    make_error_details_tool,
    make_execute_sql_tool,
    render_query_result,
    render_tool_error,
)

ANCHOR = date(2026, 11, 17)
RECURSIVE_FOREVER = (
    "WITH RECURSIVE counter(x) AS ("
    "SELECT 1 UNION ALL SELECT x + 1 FROM counter WHERE x < 100000000"
    ") SELECT COUNT(*) FROM counter"
)


@pytest.fixture(scope="module")
def database() -> MockDatabase:
    return MockDatabase.build(seed=7, months=12, reference_date=ANCHOR)


@pytest.fixture(scope="module")
def execute(database: MockDatabase, tmp_path_factory: pytest.TempPathFactory):
    """The tool the model would call, with production-shaped limits."""
    settings = Settings.from_env(
        {
            "LLM_API_KEY": "sk-test",
            "LLM_MODEL": "test-model",
            "TRACE_DIR": str(tmp_path_factory.mktemp("traces")),
        }
    )
    return make_execute_sql_tool(database, settings)


@pytest.fixture(scope="module")
def details(database: MockDatabase):
    return make_error_details_tool(database)


# --------------------------------------------------------------------------
# Successful execution
# --------------------------------------------------------------------------


def test_query_returns_typed_rows(execute) -> None:
    result = execute("SELECT COUNT(*) AS n FROM orders")
    assert isinstance(result, QueryResult)
    assert result.columns == ("n",)
    assert result.rows[0][0] > 0
    assert result.truncated is False


def test_result_carries_evidence_metadata(execute) -> None:
    result = execute("SELECT id, status FROM orders LIMIT 2")
    assert isinstance(result, QueryResult)
    assert re.fullmatch(r"q_\d{3}", result.query_id)
    assert result.duration_ms >= 0
    assert result.row_count == 2


def test_row_cap_truncates_and_says_so(execute, database: MockDatabase) -> None:
    assert database.describe("orders").table.row_count > 500
    result = execute("SELECT id FROM orders")
    assert isinstance(result, QueryResult)
    assert result.row_count == 500
    assert result.truncated is True


def test_the_executed_sql_is_the_sql_that_was_sent(execute) -> None:
    sent = "SELECT id FROM orders WHERE status = 'completed'"
    result = execute(sent)
    assert isinstance(result, QueryResult)
    assert result.sql == sent


def test_small_user_limit_is_not_widened(execute) -> None:
    result = execute("SELECT id FROM orders LIMIT 3")
    assert isinstance(result, QueryResult)
    assert result.row_count == 3
    assert result.truncated is False
    assert "LIMIT 3" in result.sql.upper()


def test_empty_result_is_success_not_error(execute) -> None:
    result = execute("SELECT id FROM orders WHERE 1 = 0")
    assert isinstance(result, QueryResult)
    assert result.rows == ()
    assert result.truncated is False


def test_aggregation_over_the_whole_table_works(execute) -> None:
    result = execute(
        """
        SELECT substr(o.order_date, 1, 7) AS month,
               ROUND(SUM(i.line_total), 2) AS revenue
        FROM orders o
        JOIN order_items i ON i.order_id = o.id
        WHERE o.status = 'completed'
        GROUP BY month
        ORDER BY month
        """
    )
    assert isinstance(result, QueryResult)
    assert result.row_count >= 3
    assert result.columns == ("month", "revenue")


def test_cte_and_window_function_work(execute) -> None:
    result = execute(
        """
        WITH monthly AS (
            SELECT substr(order_date, 1, 7) AS month, COUNT(*) AS orders
            FROM orders GROUP BY month
        )
        SELECT month, orders, RANK() OVER (ORDER BY orders DESC) AS position
        FROM monthly ORDER BY month
        """
    )
    assert isinstance(result, QueryResult)
    assert result.columns == ("month", "orders", "position")


def test_query_ids_increment(execute) -> None:
    first = execute("SELECT 1 AS a")
    second = execute("SELECT 2 AS b")
    assert isinstance(first, QueryResult) and isinstance(second, QueryResult)
    assert first.query_id != second.query_id


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE orders",
        "DELETE FROM orders",
        "UPDATE orders SET status = 'x'",
        "SELECT 1; DROP TABLE orders",
        "WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x",
        "SELECT load_extension('evil')",
        "ATTACH DATABASE 'x.db' AS x",
    ],
)
def test_unsafe_sql_is_refused_without_executing(execute, sql: str) -> None:
    result = execute(sql)
    assert isinstance(result, ToolError)
    assert result.tool == "execute_sql"
    assert result.kind == "rejected"
    assert "rule:" in result.detail


def test_refusal_names_the_rule(execute) -> None:
    result = execute("SELECT 1; DROP TABLE orders")
    assert isinstance(result, ToolError)
    assert "multi_statement" in result.detail


def test_refusals_leave_the_database_intact(execute, database: MockDatabase) -> None:
    before = database.describe("customers").table.row_count
    execute("DELETE FROM customers")
    execute("DROP TABLE customers")
    execute("UPDATE customers SET name = 'x'")
    assert database.schema().table("customers") is not None
    assert database.describe("customers").table.row_count == before


# --------------------------------------------------------------------------
# Database failures and forensics
# --------------------------------------------------------------------------


def test_invalid_column_is_a_tool_error_with_an_error_id(execute) -> None:
    result = execute("SELECT revenue FROM orders")
    assert isinstance(result, ToolError)
    assert result.kind == "invalid_query"
    assert "no such column" in result.message.lower()
    assert re.search(r"error_id: err_\d{3}", result.detail)


def test_error_details_explain_the_failure(execute, details, database: MockDatabase) -> None:
    execute("SELECT revenue FROM orders")
    error_id = database.recent_error_ids()[-1]

    text = details(error_id)
    assert isinstance(text, str)
    assert f"error_id: {error_id}" in text
    assert "invalid_query" in text
    assert "no such column" in text
    assert "SELECT revenue FROM orders" in text


def test_error_details_include_the_columns_to_fix_the_query(
    execute, details, database: MockDatabase
) -> None:
    execute("SELECT revenu FROM orders")  # typo on purpose
    text = details(database.recent_error_ids()[-1])
    assert isinstance(text, str)
    assert "referenced tables: orders" in text
    assert "columns of orders:" in text
    assert "customer_id" in text
    assert "status" in text


def test_error_details_explain_unknown_tables(execute, details, database: MockDatabase) -> None:
    execute("SELECT * FROM revenue_by_month")
    text = details(database.recent_error_ids()[-1])
    assert isinstance(text, str)
    assert "no such table" in text.lower()


def test_unknown_error_id_is_recoverable(details) -> None:
    result = details("err_999")
    assert isinstance(result, ToolError)
    assert result.kind == "invalid_input"
    assert "unknown error_id" in result.message


def test_unknown_error_id_lists_recent_ids(execute, details, database: MockDatabase) -> None:
    execute("SELECT nope FROM orders")
    recent = database.recent_error_ids()[-1]
    result = details("err_does_not_exist")
    assert isinstance(result, ToolError)
    assert recent in result.detail


def test_statement_timeout_is_enforced(database: MockDatabase, settings: Settings) -> None:
    impatient = replace(settings, statement_timeout_ms=100)
    tool = make_execute_sql_tool(database, impatient)

    result = tool(RECURSIVE_FOREVER)
    assert isinstance(result, ToolError)
    assert result.kind == "timeout"
    assert "100 ms" in result.message


def test_database_still_works_after_a_timeout(
    database: MockDatabase, settings: Settings
) -> None:
    impatient = replace(settings, statement_timeout_ms=100)
    make_execute_sql_tool(database, impatient)(RECURSIVE_FOREVER)

    recovered = make_execute_sql_tool(database, settings)("SELECT 1 AS ok")
    assert isinstance(recovered, QueryResult)
    assert recovered.rows == ((1,),)


def test_engine_blocks_writes_even_without_the_guard(database: MockDatabase) -> None:
    """Second layer: the tool is not the only thing standing between SQL and data."""
    outcome = database.execute("DELETE FROM orders", max_rows=10, timeout_ms=1_000)
    assert isinstance(outcome, QueryError)
    assert outcome.kind == "forbidden"
    assert outcome.error_id.startswith("err_")


def test_error_journal_is_bounded(database: MockDatabase) -> None:
    journal = ErrorJournal(capacity=3)
    for index in range(5):
        journal.record(kind="internal", message=f"failure {index}", sql="SELECT 1")
    assert journal.recent_ids() == ("err_003", "err_004", "err_005")
    assert journal.get("err_001") is None
    assert journal.get("err_005") is not None


def test_error_journal_rejects_a_zero_capacity() -> None:
    with pytest.raises(ValueError):
        ErrorJournal(capacity=0)


# --------------------------------------------------------------------------
# Tool contract
# --------------------------------------------------------------------------


def test_execute_tool_has_the_expected_signature(execute) -> None:
    signature = inspect.signature(execute)
    assert list(signature.parameters) == ["sql"]
    assert execute.__name__ == "execute_sql"


def test_execute_tool_documents_its_real_limits(execute) -> None:
    doc = execute.__doc__ or ""
    assert "At most 500 rows" in doc
    assert "15000 ms" in doc
    assert "20000 characters" in doc
    assert "sqlite" in doc
    assert "rejected" in doc


def test_error_details_tool_documents_its_contract(details) -> None:
    doc = details.__doc__ or ""
    assert "error_id" in doc
    assert details.__name__ == "get_query_error_details"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def test_rendering_reports_facts_before_rows() -> None:
    result = QueryResult(
        query_id="q_007",
        sql="SELECT month, revenue FROM m",
        columns=("month", "revenue"),
        rows=(("2026-09", 78084.0), ("2026-10", None)),
        truncated=False,
        duration_ms=12,
    )
    text = render_query_result(result)
    lines = text.splitlines()
    assert lines[0] == "status: ok"
    assert "query_id: q_007" in text
    assert "columns: month, revenue" in text
    assert "row_count: 2" in text
    assert "truncated: no" in text
    assert "duration_ms: 12" in text
    assert "2026-09 | 78084" in text
    assert "2026-10 | NULL" in text


def test_rendering_marks_truncation_as_actionable() -> None:
    text = render_query_result(
        QueryResult(query_id="q_1", sql="SELECT 1", columns=("a",), rows=((1,),), truncated=True)
    )
    assert "truncated: yes" in text
    assert "aggregate" in text


def test_rendering_says_an_empty_result_is_a_finding() -> None:
    text = render_query_result(
        QueryResult(query_id="q_1", sql="SELECT 1", columns=("a",), rows=())
    )
    assert "row_count: 0" in text
    assert "no rows" in text
    assert "finding" in text


def test_rendering_truncates_wide_cells() -> None:
    text = render_query_result(
        QueryResult(
            query_id="q_1",
            sql="SELECT 1",
            columns=("note",),
            rows=(("x" * 200,),),
        ),
        cell_width=10,
    )
    assert "x" * 200 not in text
    assert "…" in text


def test_rendering_collapses_multiline_sql() -> None:
    text = render_query_result(
        QueryResult(query_id="q_1", sql="SELECT\n  1\nFROM t", columns=("a",), rows=())
    )
    assert "sql: SELECT 1 FROM t" in text


def test_rendering_a_tool_error() -> None:
    text = render_tool_error(
        ToolError(
            tool="execute_sql",
            kind="rejected",
            message="2 statements were provided",
            detail="rule: multi_statement.",
        )
    )
    assert "status: error" in text
    assert "kind: rejected" in text
    assert "rule: multi_statement" in text
    assert "retryable: no" in text
