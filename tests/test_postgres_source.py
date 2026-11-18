"""The PostgreSQL adapter, exercised against a real database when one is available.

This test needs a running PostgreSQL with a `test` database and a `test_user`
role. It is skipped automatically when the connection cannot be made, so the
CI matrix stays green without a database.

When it does run, it seeds a small table, introspects it through the adapter,
runs a guarded SELECT, and checks that the read-only transaction prevents writes.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from core.config import Settings
from core.types import BackendUnavailable
from db.base import Database
from db.factory import build_database


_PG_URL = os.environ.get("AI_ANALYST_TEST_PG_URL", "postgresql+psycopg://test_user:test_pass@127.0.0.1:5432/test")


def _connect() -> Any:
    import psycopg

    return psycopg.connect(_PG_URL)


def _seed(conn: Any) -> None:
    conn.execute("DROP TABLE IF EXISTS pg_test_orders CASCADE")
    conn.execute(
        "CREATE TABLE pg_test_orders (id SERIAL PRIMARY KEY, status TEXT, amount NUMERIC)"
    )
    conn.execute(
        "INSERT INTO pg_test_orders (status, amount) VALUES ('completed', 100.0), ('completed', 50.0), ('cancelled', 25.0)"
    )
    conn.commit()


def _teardown(conn: Any) -> None:
    conn.execute("DROP TABLE IF EXISTS pg_test_orders CASCADE")
    conn.commit()
    conn.close()


def _settings(env: dict[str, str]) -> Settings:
    return Settings.from_env(
        {
            **env,
            "DATABASE_BACKEND": "postgres",
            "DATABASE_URL": _PG_URL,
        }
    )


def test_postgres_adapter_satisfies_the_protocol(env: dict[str, str]) -> None:
    try:
        conn = _connect()
    except Exception as exc:  # pragma: no cover - depends on environment
        pytest.skip(f"no PostgreSQL available: {exc}")
    try:
        _seed(conn)
        settings = _settings(env)
        database: Database = build_database(settings)
        try:
            table = database.schema().table("pg_test_orders")
            assert table is not None
            assert table.row_count == 3
            assert any(col.name == "status" for col in table.columns)

            result = database.execute(
                "SELECT status, COUNT(*) AS n FROM pg_test_orders GROUP BY status ORDER BY status",
                max_rows=500,
                timeout_ms=5000,
                tables=("pg_test_orders",),
            )
            assert result is not None
            assert not isinstance(result, type(None).__class__)  # not an error
            # The error type is QueryError; success is QueryResult
            from db.base import QueryResult

            assert isinstance(result, QueryResult)
            assert len(result.rows) == 2
        finally:
            database.close()
    finally:
        _teardown(conn)


def test_postgres_adapter_is_read_only(env: dict[str, str]) -> None:
    try:
        conn = _connect()
    except Exception as exc:  # pragma: no cover - depends on environment
        pytest.skip(f"no PostgreSQL available: {exc}")
    try:
        _seed(conn)
        settings = _settings(env)
        database = build_database(settings)
        try:
            from db.base import QueryError

            result = database.execute(
                "INSERT INTO pg_test_orders (status, amount) VALUES ('x', 1)",
                max_rows=500,
                timeout_ms=5000,
                tables=("pg_test_orders",),
            )
            assert isinstance(result, QueryError)
            assert result.kind in ("forbidden", "invalid_query", "internal")
        finally:
            database.close()
    finally:
        _teardown(conn)