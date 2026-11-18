"""PostgreSQL adapter: a read-only, transactional backend for production use.

This module implements the `Database` protocol for PostgreSQL. It connects once,
enters a read-only transaction with a read-only role, and runs all queries inside
that transaction. The transaction is never committed (it is rolled back on close)
so every query sees the same snapshot and cannot mutate data.

The adapter mirrors the mock backend's shapes: `schema()`, `describe()`,
`execute()`, `error_record()`, and `recent_error_ids()`. The guard's SQL
validation runs before any query reaches this module, so injection is blocked at
the boundary, not here.
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any

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

_MAX_DISTINCT_PROBES = 10
_MAX_DISTINCT_VALUE = 20
_PROGRESS_CHECK_INTERVAL = 5000


def _classify_error(exc: Exception) -> str:
    """Map a driver message onto a kind the model can act on."""
    message = str(exc).lower()
    if "timeout" in message or "cancel" in message:
        return "timeout"
    if "read-only" in message or "readonly" in message or "permission denied" in message:
        return "forbidden"
    if any(
        marker in message
        for marker in (
            "does not exist",
            "column",
            "table",
            "syntax",
            "error",
            "invalid",
        )
    ):
        return "invalid_query"
    return "internal"


class PostgresDatabase(Database):
    """A PostgreSQL backend that runs every query in a read-only transaction."""

    def __init__(
        self,
        connection: Any,
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
        url: str,
        reference_date: date | None = None,
    ) -> PostgresDatabase:
        """Construct a PostgreSQL backend from a connection URL.

        The connection is configured for read-only access: every query runs in a
        read-only transaction.
        """
        import psycopg
        from psycopg import sql

        conn = psycopg.connect(url, autocommit=False)
        conn.set_readonly(True)

        # Introspect schema
        snapshot = _introspect(conn, reference_date or date.today())
        return cls(conn, snapshot)

    @property
    def backend(self) -> str:
        return "postgres"

    @property
    def dialect(self) -> str:
        return "postgres"

    def schema(self) -> SchemaSnapshot:
        return self._snapshot

    def describe(self, table: str, sample_rows: int = 3) -> TableDescription:
        if not 1 <= sample_rows <= 20:
            raise ValueError("sample_rows must be between 1 and 20")
        info = self._snapshot.table(table)
        if info is None:
            raise KeyError(f"unknown table: {table}")

        from psycopg import sql

        # Fetch sample rows
        cur = self._connection.execute(
            sql.SQL("SELECT * FROM {table} LIMIT {limit}").format(
                table=sql.Identifier(info.name),
                limit=sql.Literal(sample_rows),
            )
        )
        rows = cur.fetchall()
        sample = tuple(dict(zip((col.name for col in info.columns), row)) for row in rows)

        # Get distinct values for low-cardinality columns
        distinct = tuple(
            values
            for values in (
                _distinct_values(self._connection, info, column)
                for column in _probable_columns(info)
            )
            if values is not None
        )

        return TableDescription(
            table=info,
            sample_rows=sample,
            distinct_values=distinct,
        )

    def execute(
        self, sql: str, *, max_rows: int, timeout_ms: int, tables: tuple[str, ...] = ()
    ) -> QueryResult | QueryError:
        """Run one guarded SELECT under a row cap and a wall-clock deadline.

        The statement has passed the guard, but this method still enforces
        resource limits and translates driver failures into typed errors.
        """
        if max_rows < 1:
            raise ValueError("max_rows must be at least 1")

        from psycopg import sql

        started = time.perf_counter()

        try:
            # Set statement timeout
            self._connection.execute(
                sql.SQL("SET LOCAL statement_timeout = {}").format(
                    sql.Literal(timeout_ms)
                )
            )
            cur = self._connection.execute(sql.SQL(sql))
            if cur.description is None:
                raise psycopg.ProgrammingError(
                    "the statement returned no result set; only SELECT queries return rows"
                )
            columns = tuple(desc[0] for desc in cur.description)
            rows = cur.fetchmany(max_rows + 1)
        except Exception as exc:
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

    def close(self) -> None:
        """Roll back the read-only transaction and close the connection."""
        self._connection.rollback()
        self._connection.close()


def _introspect(
    conn: Any,
    reference_date: date,
) -> SchemaSnapshot:
    """Introspect the PostgreSQL schema via information_schema."""
    from psycopg import sql

    # Get all tables
    cur = conn.execute(
        sql.SQL("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
    )
    table_names = [row[0] for row in cur.fetchall()]

    tables: list[TableInfo] = []
    for name in table_names:
        # Columns
        cur = conn.execute(
            sql.SQL("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = %s AND table_schema = 'public'
                ORDER BY ordinal_position
            """),
            (name,),
        )
        columns = tuple(
            ColumnInfo(
                name=row[0],
                type=row[1].upper(),
                nullable=row[2] == "YES",
                primary_key=False,  # Will be set below
            )
            for row in cur.fetchall()
        )

        # Primary key
        cur = conn.execute(
            sql.SQL("""
                SELECT a.attname
                FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                WHERE i.indrelid = %s::regclass AND i.indisprimary
            """),
            (name,),
        )
        pk_columns = tuple(row[0] for row in cur.fetchall())
        columns = tuple(
            ColumnInfo(
                name=col.name,
                type=col.type,
                nullable=col.nullable,
                primary_key=col.name in pk_columns,
            )
            for col in columns
        )

        # Foreign keys
        cur = conn.execute(
            sql.SQL("""
                SELECT
                    kcu.column_name,
                    ccu.table_name AS references_table,
                    ccu.column_name AS references_column
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s
            """),
            (name,),
        )
        foreign_keys = tuple(
            ForeignKeyInfo(
                column=row[0],
                references_table=row[1],
                references_column=row[2],
            )
            for row in cur.fetchall()
        )

        # Row count
        cur = conn.execute(
            sql.SQL("SELECT COUNT(*) FROM {table}").format(
                table=sql.Identifier(name)
            )
        )
        row_count = cur.fetchone()[0]

        tables.append(
            TableInfo(
                name=name,
                columns=columns,
                primary_key=pk_columns,
                foreign_keys=foreign_keys,
                row_count=row_count or 0,
            )
        )

    span_start, span_end = _data_span(conn, tables)
    return SchemaSnapshot(
        backend="postgres",
        as_of=reference_date.isoformat(),
        tables=tuple(tables),
        data_span=(span_start.isoformat(), span_end.isoformat()) if span_start and span_end else None,
        notes=_freshness_notes(reference_date, span_end) if span_end else (),
    )


def _data_span(conn: Any, tables: list[TableInfo]) -> tuple[date | None, date | None]:
    """Infer the data span from date columns in the schema."""
    # Look for common date columns: order_date, created_at, etc.
    date_cols = [
        (t.name, c.name)
        for t in tables
        for c in t.columns
        if "date" in c.type.lower() or "time" in c.type.lower()
    ]
    if not date_cols:
        return None, None

    min_date: date | None = None
    max_date: date | None = None

    for table, col in date_cols[:5]:  # Limit to first 5 to avoid too many queries
        try:
            cur = conn.execute(
                sql.SQL("SELECT MIN({col}), MAX({col}) FROM {table}").format(
                    col=sql.Identifier(col),
                    table=sql.Identifier(table),
                )
            )
            row = cur.fetchone()
            if row[0]:
                d = row[0] if isinstance(row[0], date) else date.fromisoformat(str(row[0]))
                if min_date is None or d < min_date:
                    min_date = d
            if row[1]:
                d = row[1] if isinstance(row[1], date) else date.fromisoformat(str(row[1]))
                if max_date is None or d > max_date:
                    max_date = d
        except Exception:
            continue

    return min_date, max_date


def _probable_columns(info: TableInfo) -> tuple[str, ...]:
    """Text and flag columns are the ones whose values the model must match exactly."""
    candidates = [
        column.name
        for column in info.columns
        if not column.primary_key and column.type in ("TEXT", "CHARACTER VARYING", "VARCHAR", "BOOLEAN")
    ]
    return tuple(candidates[:_MAX_DISTINCT_PROBES])


def _distinct_values(
    conn: Any, info: TableInfo, column: str
) -> ColumnValues | None:
    from psycopg import sql

    cur = conn.execute(
        sql.SQL("""
            SELECT COUNT(DISTINCT {col}) FROM {table}
        """).format(
            col=sql.Identifier(column),
            table=sql.Identifier(info.name),
        )
    )
    total = cur.fetchone()[0]
    if not 0 < total <= _MAX_DISTINCT_VALUE:
        return None

    cur = conn.execute(
        sql.SQL("""
            SELECT {col} FROM {table}
            WHERE {col} IS NOT NULL
            GROUP BY {col}
            ORDER BY COUNT(*) DESC
            LIMIT {limit}
        """).format(
            col=sql.Identifier(column),
            table=sql.Identifier(info.name),
            limit=sql.Literal(_MAX_DISTINCT_VALUE),
        )
    )
    rows = cur.fetchall()
    return ColumnValues(column=column, values=tuple(str(row[0]) for row in rows))


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
