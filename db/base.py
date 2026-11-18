"""Database adapter shapes and the backend protocol.

Capability-local types: these describe an external system, they are not
orchestration contracts (those live in `core/types.py`). The same shapes are
produced by the mock and PostgreSQL adapters, which is what lets schema
inspection and SQL execution be backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False


@dataclass(frozen=True)
class ForeignKeyInfo:
    column: str
    references_table: str
    references_column: str


@dataclass(frozen=True)
class TableInfo:
    name: str
    columns: tuple[ColumnInfo, ...]
    primary_key: tuple[str, ...] = ()
    foreign_keys: tuple[ForeignKeyInfo, ...] = ()
    row_count: int = 0


@dataclass(frozen=True)
class SchemaSnapshot:
    """Everything a planner needs to know about the shape and freshness of data."""

    backend: str
    as_of: str
    tables: tuple[TableInfo, ...]
    data_span: tuple[str, str] | None = None
    notes: tuple[str, ...] = ()

    def table(self, name: str) -> TableInfo | None:
        lowered = name.strip().lower()
        for table in self.tables:
            if table.name.lower() == lowered:
                return table
        return None

    @property
    def table_names(self) -> tuple[str, ...]:
        return tuple(table.name for table in self.tables)


@dataclass(frozen=True)
class ColumnValues:
    """Values of a low-cardinality column, e.g. every order status that exists."""

    column: str
    values: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TableDescription:
    """Full detail for one table: structure, real sample rows, value domains."""

    table: TableInfo
    sample_rows: tuple[dict[str, Any], ...] = ()
    distinct_values: tuple[ColumnValues, ...] = ()


@dataclass(frozen=True)
class QueryResult:
    """Rows returned by an executed statement, already capped and timed."""

    query_id: str
    sql: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    truncated: bool = False
    duration_ms: int = 0

    @property
    def row_count(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class QueryError:
    """A database failure, addressed by `error_id` for later forensics (D-06).

    `kind` is one of: invalid_query (bad table/column/syntax), timeout,
    forbidden (refused by the engine), internal.
    """

    error_id: str
    kind: str
    message: str
    sql: str
    tables: tuple[str, ...] = ()
    driver_detail: str = ""
    position: int | None = None
    retryable: bool = False


class ErrorJournal:
    """Bounded, id-addressed store of recent execution failures.

    Living in the adapter keeps the tool layer stateless: `execute_sql` hands the
    model a short message plus an id, and `get_query_error_details` trades that id
    for the full driver payload.
    """

    def __init__(self, capacity: int = 20) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
        self._records: dict[str, QueryError] = {}
        self._order: list[str] = []
        self._counter = 0

    def record(
        self,
        *,
        kind: str,
        message: str,
        sql: str,
        tables: tuple[str, ...] = (),
        driver_detail: str = "",
        position: int | None = None,
        retryable: bool = False,
    ) -> QueryError:
        self._counter += 1
        error = QueryError(
            error_id=f"err_{self._counter:03d}",
            kind=kind,
            message=message,
            sql=sql,
            tables=tables,
            driver_detail=driver_detail,
            position=position,
            retryable=retryable,
        )
        self._records[error.error_id] = error
        self._order.append(error.error_id)
        while len(self._order) > self._capacity:
            self._records.pop(self._order.pop(0), None)
        return error

    def get(self, error_id: str) -> QueryError | None:
        return self._records.get((error_id or "").strip())

    def recent_ids(self) -> tuple[str, ...]:
        return tuple(self._order)


class Database(Protocol):
    """Read-only access to one database. Implementations are immutable after build."""

    @property
    def backend(self) -> str: ...

    @property
    def dialect(self) -> str: ...

    def schema(self) -> SchemaSnapshot: ...

    def describe(self, table: str, sample_rows: int = 3) -> TableDescription: ...

    def execute(
        self, sql: str, *, max_rows: int, timeout_ms: int, tables: tuple[str, ...] = ()
    ) -> QueryResult | QueryError: ...

    def error_record(self, error_id: str) -> QueryError | None: ...

    def recent_error_ids(self) -> tuple[str, ...]: ...
