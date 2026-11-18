"""SQL tools: the only path from a model to the database.

Two tools with different jobs:

- `execute_sql` guards, executes, and returns typed data (not prose), so the
  orchestrator can record evidence without parsing text.
- `get_query_error_details` exchanges an `error_id` for the full failure payload
  plus the columns of the tables involved, so a model can repair a query instead
  of guessing.

Docstrings are the schemas the model sees, and they state the real limits
(T-01, T-06). The rendered text is what the model reads; the transport decides
how to print it (P-01).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.config import Settings
from core.types import ToolError
from db.base import Database, QueryError, QueryResult
from db.guard import GuardRejection, guard

ExecuteSqlTool = Callable[[str], QueryResult | ToolError]
ErrorDetailsTool = Callable[[str], str | ToolError]

_CELL_WIDTH = 40
_DETAIL_WIDTH = 500

_EXECUTE_SQL_DOC = """Execute exactly one read-only SQL query and return its rows.

The database is read-only: you cannot change data, create objects, or read files.
Writes, DDL, PRAGMA, ATTACH, and multiple statements are refused before execution.

Dialect: {dialect}. CTEs, joins, subqueries, window functions, and aggregates are fine.

Args:
    sql: one SELECT statement, e.g. "SELECT status, COUNT(*) FROM orders GROUP BY status".
         Do not include a trailing semicolon or a second statement.

Returns:
    On success: a result whose text form lists query_id, columns, row_count,
    truncated, duration_ms, and the rows. An empty row list is a valid finding.
    At most {max_rows} rows are returned; when `truncated` is yes, aggregate in SQL
    instead of fetching raw rows.
    Limits: {timeout_ms} ms per statement, {max_length} characters of SQL.
    On refusal: a ToolError with kind "rejected"; nothing was executed.
    On failure: a ToolError with kind "invalid_query", "timeout", "forbidden", or
    "internal"; its detail carries an error_id for get_query_error_details.
"""

_ERROR_DETAILS_DOC = """Explain a failed query in full: the driver message, the SQL that failed,
and the exact columns of every table it referenced.

Call this after execute_sql returns an error, when the short message is not enough
to fix the query. Use it instead of guessing column names.

Args:
    error_id: the id reported by a failed execute_sql, e.g. "err_002".

Returns:
    Plain text: the error kind and message, the failing SQL, the columns of the
    referenced tables (with types and keys), and the raw driver detail.
    On an unknown id: a ToolError with kind "invalid_input" listing recent ids.
"""


def make_execute_sql_tool(database: Database, settings: Settings) -> ExecuteSqlTool:
    """Bind execution to one database and one set of limits."""

    def execute_sql(sql: str) -> QueryResult | ToolError:
        guarded = guard(
            sql,
            dialect=database.dialect,
            max_length=settings.max_sql_length,
        )
        if isinstance(guarded, GuardRejection):
            return ToolError(
                tool="execute_sql",
                kind="rejected",
                message=guarded.message,
                detail=f"rule: {guarded.rule}. {guarded.hint}",
            )

        outcome = database.execute(
            guarded.sql,
            max_rows=settings.max_rows,
            timeout_ms=settings.statement_timeout_ms,
            tables=guarded.tables,
        )
        if isinstance(outcome, QueryError):
            return ToolError(
                tool="execute_sql",
                kind=outcome.kind,
                message=outcome.message,
                retryable=outcome.retryable,
                detail=(
                    f"error_id: {outcome.error_id}. "
                    "call get_query_error_details with this id for the driver message "
                    "and the columns of the tables involved"
                ),
                error_id=outcome.error_id,
            )
        return outcome

    execute_sql.__doc__ = _EXECUTE_SQL_DOC.format(
        dialect=database.dialect,
        max_rows=settings.max_rows,
        timeout_ms=settings.statement_timeout_ms,
        max_length=settings.max_sql_length,
    )
    return execute_sql


def make_error_details_tool(database: Database) -> ErrorDetailsTool:
    """Bind failure forensics to the adapter that recorded the ids."""

    def get_query_error_details(error_id: str) -> str | ToolError:
        record = database.error_record(error_id)
        if record is None:
            known = database.recent_error_ids()
            return ToolError(
                tool="get_query_error_details",
                kind="invalid_input",
                message=f"unknown error_id {(error_id or '').strip()!r}",
                detail=(
                    f"recent error ids: {', '.join(known)}"
                    if known
                    else "no query has failed yet in this run"
                ),
            )
        return _render_error_record(record, database)

    get_query_error_details.__doc__ = _ERROR_DETAILS_DOC
    return get_query_error_details


_SAMPLE_ROWS = 5


def render_query_result(result: QueryResult, *, cell_width: int = _CELL_WIDTH) -> str:
    """The text form a model reads. Facts first, rows last, no markup dialects.

    Only a small sample of rows is sent to the model. The full result is kept
    in the evidence trail for later citation; the model does not need hundreds
    of rows to form a finding.
    """
    truncated = (
        "yes - cut at the row cap, so aggregate in SQL instead of fetching raw rows"
        if result.truncated
        else "no"
    )
    lines = [
        "status: ok",
        f"query_id: {result.query_id}",
        f"sql: {_one_line(result.sql)}",
        f"columns: {', '.join(result.columns)}",
        f"row_count: {result.row_count}",
        f"truncated: {truncated}",
        f"duration_ms: {result.duration_ms}",
    ]
    if result.rows:
        sample = result.rows[:_SAMPLE_ROWS]
        lines.append(
            f"sample (first {len(sample)} of {result.row_count} rows):"
        )
        lines.append(" | ".join(result.columns))
        for row in sample:
            lines.append(
                " | ".join(_cell(value, cell_width) for value in row)
            )
        if result.row_count > _SAMPLE_ROWS:
            lines.append(
                f"({result.row_count - _SAMPLE_ROWS} more rows - use aggregate "
                "functions in SQL to summarize instead of fetching raw rows)"
            )
    else:
        lines.append("(no rows; an empty result is a finding, not an error)")
    return "\n".join(lines)


def render_result_preview(
    result: QueryResult, *, rows: int = 5, chars: int = 800
) -> str:
    """A bounded excerpt of the rows, for the evidence trail.

    Not for the model: the model already received the full (capped) result. This
    is what a later step reads to cite a number without re-running the query, so
    it is small and deterministic.
    """
    if rows <= 0 or not result.rows:
        return ""
    lines = [" | ".join(result.columns)]
    lines.extend(
        " | ".join(_cell(value, _CELL_WIDTH) for value in row)
        for row in result.rows[:rows]
    )
    omitted = result.row_count - (len(lines) - 1)
    if omitted > 0:
        lines.append(f"({omitted} more rows)")
    text = "\n".join(lines)
    return text if len(text) <= chars else text[: chars - 1] + "…"


def render_tool_error(error: ToolError) -> str:
    """The text form of a recoverable failure."""
    lines = [
        "status: error",
        f"tool: {error.tool}",
        f"kind: {error.kind}",
        f"message: {_one_line(error.message)}",
    ]
    if error.detail:
        lines.append(f"detail: {error.detail}")
    lines.append(f"retryable: {'yes' if error.retryable else 'no'}")
    return "\n".join(lines)


def _render_error_record(record: QueryError, database: Database) -> str:
    lines = [
        f"error_id: {record.error_id}",
        f"kind: {record.kind}",
        f"message: {_one_line(record.message)}",
        f"sql: {_one_line(record.sql)}",
    ]
    if record.position is not None:
        lines.append(f"position: character {record.position} of the statement")
    lines.append(f"retryable: {'yes' if record.retryable else 'no'}")

    if not record.tables:
        lines.append("referenced tables: none")
    else:
        lines.append(f"referenced tables: {', '.join(record.tables)}")
        snapshot = database.schema()
        for name in record.tables:
            table = snapshot.table(name)
            if table is None:
                continue
            lines.append(f"columns of {table.name}:")
            for column in table.columns:
                flags = "  PK" if column.primary_key else ""
                lines.append(f"  {column.name} {column.type}{flags}")

    if record.driver_detail:
        lines.append("driver detail:")
        lines.append(f"  {_one_line(record.driver_detail)[:_DETAIL_WIDTH]}")
    return "\n".join(lines)


def _cell(value: Any, width: int) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def _one_line(text: str, limit: int = _DETAIL_WIDTH) -> str:
    collapsed = " ".join((text or "").split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"
