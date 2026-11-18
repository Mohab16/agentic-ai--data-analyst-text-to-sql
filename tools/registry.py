"""Tool dispatch: what a model may call, what the arguments mean, and the budgets.

Three tools, one entry point. `Toolset.call` never raises: a malformed call, an
unknown tool, and an exhausted budget all come back as a rendered failure the model
can read, because a crashed loop teaches it nothing.

Two budgets live here rather than in the agent, because this is where the counter
is:

- `MAX_QUERIES` bounds statements that reached the database. A guard refusal never
  touches the database, so it is charged to the failure allowance instead.
- `MAX_QUERY_RETRIES` bounds failures: one attempt plus that many retries. Without
  it a model can spend the whole query budget resending the same broken statement.

Every executed statement also becomes a `QueryRecord`: the evidence id the model
cites, the SQL that actually ran, and a bounded preview of the rows.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.config import Settings
from core.types import QueryRecord, ToolError
from db.base import Database, QueryResult
from llm.base import ToolCall, ToolSpec
from tools.schema import make_inspect_schema_tool
from tools.sql import (
    make_error_details_tool,
    make_execute_sql_tool,
    render_query_result,
    render_result_preview,
    render_tool_error,
)

EXECUTE_SQL = "execute_sql"
INSPECT_SCHEMA = "inspect_schema"
ERROR_DETAILS = "get_query_error_details"

_NOT_RUN_HINT = (
    "no further queries are accepted in this run: summarize the evidence you "
    "already have, and say what you could not check"
)


@dataclass(frozen=True)
class ToolOutcome:
    """What the model reads, and the evidence the run keeps."""

    text: str
    ok: bool
    record: QueryRecord | None = None


def build_toolset(database: Database, settings: Settings) -> Toolset:
    """One toolset per run: it holds the budgets and the evidence records."""
    return Toolset(database, settings)


class Toolset:
    """The three tools bound to one database, with the limits of one run."""

    def __init__(self, database: Database, settings: Settings) -> None:
        self._settings = settings
        self._execute = make_execute_sql_tool(database, settings)
        self._inspect = make_inspect_schema_tool(database)
        self._details = make_error_details_tool(database)
        self._specs = _build_specs(settings)
        self._handlers: dict[str, Callable[[dict[str, Any]], ToolOutcome]] = {
            EXECUTE_SQL: self._execute_sql,
            INSPECT_SCHEMA: self._inspect_schema,
            ERROR_DETAILS: self._error_details,
        }

        self.records: list[QueryRecord] = []
        self.queries_used = 0
        self.failures = 0
        self.refused = 0
        self._rejections = 0
        self._results: dict[str, QueryResult] = {}

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return self._specs

    @property
    def failure_allowance(self) -> int:
        """One attempt plus `MAX_QUERY_RETRIES` retries."""
        return 1 + self._settings.max_query_retries

    @property
    def budget_exhausted(self) -> bool:
        return (
            self.queries_used >= self._settings.max_queries
            or self.failures >= self.failure_allowance
        )

    def call(self, call: ToolCall) -> ToolOutcome:
        """Dispatch one tool call. Never raises: failures are values (T-03)."""
        if call.parse_error:
            return self._bad_call(
                call.name or "unknown",
                f"the arguments could not be read: {call.parse_error}",
            )
        handler = self._handlers.get(call.name)
        if handler is None:
            return self._bad_call(
                call.name or "unknown",
                f"no such tool; available: {', '.join(self._handlers)}",
            )
        return handler(call.arguments)

    # ------------------------------------------------------------------
    # execute_sql
    # ------------------------------------------------------------------

    def _execute_sql(self, arguments: dict[str, Any]) -> ToolOutcome:
        sql = arguments.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            return self._bad_call(
                EXECUTE_SQL, 'execute_sql needs a non-empty string argument "sql"'
            )

        blocked = self._budget_refusal(sql)
        if blocked is not None:
            return blocked

        outcome = self._execute(sql)
        rejected = isinstance(outcome, ToolError) and outcome.kind == "rejected"
        if not rejected:
            self.queries_used += 1

        if isinstance(outcome, ToolError):
            self.failures += 1
            if rejected:
                self._rejections += 1
            record = QueryRecord(
                id=outcome.error_id or f"r{self._rejections}",
                sql=sql,
                status="error",
                error_kind=outcome.kind,
                error_message=outcome.message,
            )
        else:
            self._results[outcome.query_id] = outcome
            record = QueryRecord(
                id=outcome.query_id,
                sql=outcome.sql,
                status="ok",
                row_count=outcome.row_count,
                duration_ms=outcome.duration_ms,
                truncated=outcome.truncated,
                columns=outcome.columns,
                preview=render_result_preview(
                    outcome, rows=self._settings.trace_sample_rows
                ),
            )

        self.records.append(record)
        text = (
            render_tool_error(outcome)
            if isinstance(outcome, ToolError)
            else render_query_result(outcome)
        )
        return ToolOutcome(text=text, ok=not isinstance(outcome, ToolError), record=record)

    def _budget_refusal(self, sql: str) -> ToolOutcome | None:
        if self.queries_used >= self._settings.max_queries:
            return self._not_run(
                sql,
                f"query budget exhausted: {self.queries_used} of "
                f"{self._settings.max_queries} queries used",
            )
        if self.failures >= self.failure_allowance:
            return self._not_run(
                sql,
                f"too many failed queries: {self.failures} of "
                f"{self.failure_allowance} allowed",
            )
        return None

    def _not_run(self, sql: str, message: str) -> ToolOutcome:
        self.refused += 1
        record = QueryRecord(
            id=f"x{self.refused}",
            sql=sql,
            status="error",
            error_kind="not_run",
            error_message=message,
        )
        self.records.append(record)
        return ToolOutcome(
            text=render_tool_error(
                ToolError(
                    tool=EXECUTE_SQL,
                    kind="not_run",
                    message=message,
                    detail=_NOT_RUN_HINT,
                )
            ),
            ok=False,
            record=record,
        )

    # ------------------------------------------------------------------
    # inspect_schema / get_query_error_details
    # ------------------------------------------------------------------

    def _inspect_schema(self, arguments: dict[str, Any]) -> ToolOutcome:
        table = arguments.get("table")
        if not isinstance(table, str) or not table.strip():
            return self._bad_call(
                INSPECT_SCHEMA, 'inspect_schema needs a non-empty string argument "table"'
            )
        return _as_outcome(self._inspect(table))

    def _error_details(self, arguments: dict[str, Any]) -> ToolOutcome:
        error_id = arguments.get("error_id")
        if not isinstance(error_id, str) or not error_id.strip():
            return self._bad_call(
                ERROR_DETAILS,
                'get_query_error_details needs a non-empty string argument "error_id"',
            )
        return _as_outcome(self._details(error_id))

    def _bad_call(self, tool: str, message: str) -> ToolOutcome:
        return ToolOutcome(
            text=render_tool_error(
                ToolError(tool=tool, kind="invalid_arguments", message=message)
            ),
            ok=False,
        )

    def numbers_by_query(self) -> dict[str, tuple[float, ...]]:
        """Numeric values from each successful query, for evidence checking.

        Every finite number that appears in a result row, keyed by query id, so a
        claim's numbers can be checked against the rows that produced them.
        """
        index: dict[str, list[float]] = {}
        for query_id, result in self._results.items():
            values: list[float] = []
            for row in result.rows:
                for value in row:
                    if isinstance(value, bool):
                        continue
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        values.append(float(value))
            index[query_id] = tuple(values)
        return index


def _as_outcome(result: str | ToolError) -> ToolOutcome:
    if isinstance(result, ToolError):
        return ToolOutcome(text=render_tool_error(result), ok=False)
    return ToolOutcome(text=result, ok=True)


def _build_specs(settings: Settings) -> tuple[ToolSpec, ...]:
    return (
        ToolSpec(
            name=EXECUTE_SQL,
            description=(
                "Run exactly one read-only SELECT and return its rows. Writes, DDL, "
                "PRAGMA, ATTACH, and multiple statements are refused before "
                f"execution. At most {settings.max_rows} rows; the statement is "
                f"cancelled after {settings.statement_timeout_ms} ms. On failure the "
                "message carries an error_id for get_query_error_details. A zero-row "
                "result is a valid finding."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": (
                            "one SELECT statement, no trailing semicolon "
                            f"(at most {settings.max_sql_length} characters)"
                        ),
                    }
                },
                "required": ["sql"],
            },
        ),
        ToolSpec(
            name=INSPECT_SCHEMA,
            description=(
                "Full detail for one table: columns with types and keys, "
                "low-cardinality value domains, and real sample rows. Use it when the "
                "schema digest is not specific enough to write a column name."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "table name, e.g. order_items",
                    }
                },
                "required": ["table"],
            },
        ),
        ToolSpec(
            name=ERROR_DETAILS,
            description=(
                "Explain a failed execute_sql call in full: the driver message, the "
                "failing SQL, and the columns of every table it referenced. Use it "
                "instead of guessing column names."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "error_id": {
                        "type": "string",
                        "description": 'the id from the failure, e.g. "err_002"',
                    }
                },
                "required": ["error_id"],
            },
        ),
    )
