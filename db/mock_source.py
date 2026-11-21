"""Mock backend: SQLite in memory, seeded once, strictly read-only afterwards.

The engine enforces read-only independently of any SQL guard: an authorizer
denies every write, DDL, PRAGMA, ATTACH, and transaction action. The mock
dataset is immutable after build, so the schema snapshot is introspected once
and cached.

Never opens files, never touches the network, and is built on explicit request
only (G-04).
"""

from __future__ import annotations

import sqlite3
import time
from datetime import date

from db.base import (
    ColumnInfo,
    ColumnValues,
    Database,
    ErrorJournal,
    ForeignKeyInfo,
    QueryError,
    QueryResult,
    SchemaSnapshot,
    TableDescription,
    TableInfo,
)
from db.seed import SeedConfig, build_dataset, create_schema, data_span, insert_dataset

# Every action that could change the database, the connection, or its schema.
# Includes INSERT OR REPLACE via SQLITE_INSERT.
_DENIED_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX,
        sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
        sqlite3.SQLITE_CREATE_TEMP_VIEW,
        sqlite3.SQLITE_CREATE_TRIGGER,
        sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_CREATE_VTABLE,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_INDEX,
        sqlite3.SQLITE_DROP_TEMP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_TRIGGER,
        sqlite3.SQLITE_DROP_TEMP_VIEW,
        sqlite3.SQLITE_DROP_TRIGGER,
        sqlite3.SQLITE_DROP_VIEW,
        sqlite3.SQLITE_DROP_VTABLE,
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_REINDEX,
        sqlite3.SQLITE_ANALYZE,
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_PRAGMA,
        sqlite3.SQLITE_TRANSACTION,
        sqlite3.SQLITE_SAVEPOINT,
    }
)

_MAX_DISTINCT_PROBES = 10
_MAX_DISTINCT_VALUE = 20

# How often SQLite calls back while executing; fine-grained enough to abort a
# runaway query close to the deadline, coarse enough to stay cheap.
_PROGRESS_STEPS = 5_000


def _authorizer(
    action: int, _arg1: str | None, _arg2: str | None, _dbname: str | None, _trigger: str | None
) -> int:
    if action in _DENIED_ACTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _classify_error(exc: sqlite3.Error) -> str:
    """Map a driver message onto a kind the model can act on.
    
    Uses lowercase comparison for locale-independence.
    """
    message = str(exc).lower()
    if "interrupted" in message or "timeout" in message or "cancel" in message:
        return "timeout"
    if "not authorized" in message or "readonly" in message:
        return "forbidden"
    if any(
        marker in message
        for marker in (
            "no such table",
            "no such column",
            "no such function",
            "syntax error",
            "incomplete input",
            "ambiguous column",
            "misuse of aggregate",
            "datatype mismatch",
            "wrong number of arguments",
            "only one statement",
        )
    ):
        return "invalid_query"
    return "internal"


class MockDatabase(Database):
    """A seeded in-memory SQLite warehouse, read-only by construction."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        snapshot: SchemaSnapshot,
    ) -> None:
        self._connection = connection
        self._snapshot = snapshot
        self._errors = ErrorJournal()
        self._query_counter = 0

    @classmethod
    def build(
        cls,
        *,
        seed: int = 42,
        months: int = 24,
        reference_date: date | None = None,
    ) -> MockDatabase:
        config = SeedConfig(
            seed=seed,
            months=months,
            reference_date=reference_date or date.today(),
        )
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        create_schema(connection)
        insert_dataset(connection, build_dataset(config))

        span_start, span_end = data_span(config)
        snapshot = _introspect(
            connection,
            as_of=config.reference_date.isoformat(),
            data_span=(span_start.isoformat(), span_end.isoformat()),
            notes=_freshness_notes(config.reference_date, span_end),
        )
        # Lock the connection after seeding: nothing below this line can write.
        connection.set_authorizer(_authorizer)
        return cls(connection, snapshot)

    @property
    def backend(self) -> str:
        return "mock"

    @property
    def dialect(self) -> str:
        return "sqlite"

    @property
    def connection(self) -> sqlite3.Connection:
        """The read-only connection. The SQL executor and security tests use it.

        It is locked by the authorizer installed in `build`, so writes, DDL,
        PRAGMA, ATTACH, and explicit transactions all fail at engine level.
        """
        return self._connection

    def schema(self) -> SchemaSnapshot:
        return self._snapshot

    def describe(self, table: str, sample_rows: int = 3) -> TableDescription:
        if not 1 <= sample_rows <= 20:
            raise ValueError("sample_rows must be between 1 and 20")
        info = self._snapshot.table(table)
        if info is None:
            raise KeyError(f"unknown table: {table}")

        quoted = _quote(info.name)
        rows = self._connection.execute(
            f"SELECT * FROM {quoted} LIMIT ?", (sample_rows,)
        ).fetchall()
        return TableDescription(
            table=info,
            sample_rows=tuple({key: row[key] for key in row.keys()} for row in rows),
            distinct_values=tuple(
                values
                for values in (
                    _distinct_values(self._connection, info, column)
                    for column in _probable_columns(info)
                )
                if values is not None
            ),
        )

    def execute(
        self, sql: str, *, max_rows: int, timeout_ms: int, tables: tuple[str, ...] = ()
    ) -> QueryResult | QueryError:
        """Run one statement under a row cap and a wall-clock deadline.

        The engine lock already makes writes impossible; this method enforces
        resource limits and translates driver failures into typed errors. The
        statement it receives has passed the guard, but it does not depend on
        that: an unguarded write still fails at the authorizer.
        """
        if max_rows < 1:
            raise ValueError("max_rows must be at least 1")

        started = time.perf_counter()
        deadline = started + timeout_ms / 1000.0

        def expired() -> int:
            return 1 if time.perf_counter() >= deadline else 0

        try:
            self._connection.set_progress_handler(expired, _PROGRESS_STEPS)
            cursor = self._connection.execute(sql)
            if cursor.description is None:
                raise sqlite3.ProgrammingError(
                    "the statement returned no result set; only SELECT queries return rows"
                )
            columns = tuple(description[0] for description in cursor.description)
            rows = cursor.fetchmany(max_rows + 1)
        except sqlite3.Error as exc:
            kind = _classify_error(exc)
            message = str(exc)
            if kind == "timeout":
                message = (
                    f"the statement exceeded the {timeout_ms} ms limit and was cancelled; "
                    "filter more, aggregate in SQL, or narrow the time range"
                )
            return self._errors.record(
                kind=kind,
                message=message,
                sql=sql,
                tables=tables,
                driver_detail=f"{type(exc).__name__}: {exc}",
            )
        finally:
            self._connection.set_progress_handler(None, 0)

        duration_ms = int((time.perf_counter() - started) * 1000)
        truncated = len(rows) > max_rows
        if truncated:
            rows = rows[:max_rows]

        self._query_counter += 1
        return QueryResult(
            query_id=f"q_{self._query_counter:03d}",
            sql=sql,
            columns=columns,
            rows=tuple(tuple(row) for row in rows),
            truncated=truncated,
            duration_ms=duration_ms,
        )

    def error_record(self, error_id: str) -> QueryError | None:
        return self._errors.get(error_id)

    def recent_error_ids(self) -> tuple[str, ...]:
        return self._errors.recent_ids()


def _introspect(
    connection: sqlite3.Connection,
    *,
    as_of: str,
    data_span: tuple[str, str] | None,
    notes: tuple[str, ...],
) -> SchemaSnapshot:
    table_rows = connection.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()

    tables: list[TableInfo] = []
    for (name,) in table_rows:
        quoted = _quote(name)
        columns = tuple(
            ColumnInfo(
                name=column["name"],
                type=(column["type"] or "TEXT").upper(),
                nullable=not column["notnull"] and not column["pk"],
                primary_key=bool(column["pk"]),
            )
            for column in connection.execute(f"PRAGMA table_info({quoted})").fetchall()
        )
        foreign_keys = tuple(
            ForeignKeyInfo(
                column=row["from"],
                references_table=row["table"],
                references_column=row["to"],
            )
            for row in connection.execute(f"PRAGMA foreign_key_list({quoted})").fetchall()
        )
        row_count = connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
        tables.append(
            TableInfo(
                name=name,
                columns=columns,
                primary_key=tuple(column.name for column in columns if column.primary_key),
                foreign_keys=foreign_keys,
                row_count=row_count,
            )
        )

    return SchemaSnapshot(
        backend="mock",
        as_of=as_of,
        tables=tuple(tables),
        data_span=data_span,
        notes=notes,
    )


def _probable_columns(info: TableInfo) -> tuple[str, ...]:
    """Text and flag columns are the ones whose values the model must match exactly."""
    candidates = [
        column.name
        for column in info.columns
        if not column.primary_key and column.type in ("TEXT", "INTEGER")
    ]
    return tuple(candidates[:_MAX_DISTINCT_PROBES])


def _distinct_values(
    connection: sqlite3.Connection, info: TableInfo, column: str
) -> ColumnValues | None:
    quoted_table = _quote(info.name)
    quoted_column = _quote(column)
    (total,) = connection.execute(
        f"SELECT COUNT(DISTINCT {quoted_column}) FROM {quoted_table}"
    ).fetchone()
    if not 0 < total <= _MAX_DISTINCT_VALUE:
        return None
    rows = connection.execute(
        f"SELECT {quoted_column} FROM {quoted_table} "
        f"WHERE {quoted_column} IS NOT NULL "
        f"GROUP BY {quoted_column} ORDER BY COUNT(*) DESC LIMIT {_MAX_DISTINCT_VALUE}"
    ).fetchall()
    return ColumnValues(column=column, values=tuple(str(row[0]) for row in rows))


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _freshness_notes(reference_date: date, data_end: date) -> tuple[str, ...]:
    """Tell the planner that the current month is not a whole month (D-07)."""
    notes: list[str] = []
    if data_end < reference_date and data_end.month == reference_date.month:
        notes.append(
            f"the current month is partial: data ends {data_end.isoformat()}, "
            f"so month-to-date must be compared with the same span, not with a full month"
        )
    notes.append("amounts are in the company's reporting currency, stored as decimals")
    return tuple(notes)
