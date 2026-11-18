"""The mock adapter: introspection, detail, and engine-level read-only access."""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from db.base import ColumnInfo, TableInfo
from db.mock_source import MockDatabase

ANCHOR = date(2026, 11, 17)


@pytest.fixture(scope="module")
def database() -> MockDatabase:
    return MockDatabase.build(seed=42, months=24, reference_date=ANCHOR)


def table(snapshot_table: TableInfo | None) -> TableInfo:
    assert snapshot_table is not None
    return snapshot_table


# --------------------------------------------------------------------------
# Schema snapshot
# --------------------------------------------------------------------------


def test_backend_is_mock(database: MockDatabase) -> None:
    assert database.backend == "mock"


def test_snapshot_lists_the_business_tables(database: MockDatabase) -> None:
    names = set(database.schema().table_names)
    assert names == {
        "customers",
        "products",
        "orders",
        "order_items",
        "payments",
        "returns",
    }


def test_snapshot_row_counts_match_the_data(database: MockDatabase) -> None:
    orders = table(database.schema().table("orders"))
    assert orders.row_count > 3_000
    by_name = {t.name: t for t in database.schema().tables}
    assert by_name["order_items"].row_count > by_name["orders"].row_count


def test_columns_carry_types_and_keys(database: MockDatabase) -> None:
    orders = table(database.schema().table("orders"))
    columns = {column.name: column for column in orders.columns}
    assert columns["id"].primary_key is True
    assert columns["id"].type == "INTEGER"
    assert columns["status"].type == "TEXT"
    assert columns["status"].nullable is False
    assert orders.primary_key == ("id",)


def test_foreign_keys_are_discovered(database: MockDatabase) -> None:
    order_items = table(database.schema().table("order_items"))
    targets = {(fk.column, fk.references_table, fk.references_column) for fk in order_items.foreign_keys}
    assert ("order_id", "orders", "id") in targets
    assert ("product_id", "products", "id") in targets


def test_table_lookup_is_case_insensitive(database: MockDatabase) -> None:
    assert database.schema().table("ORDERS") is not None
    assert database.schema().table("  Orders ") is not None
    assert database.schema().table("nope") is None


def test_snapshot_records_freshness(database: MockDatabase) -> None:
    snapshot = database.schema()
    assert snapshot.as_of == "2026-11-17"
    assert snapshot.data_span == ("2024-12-01", "2026-11-16")
    assert any("partial" in note for note in snapshot.notes)


def test_column_info_defaults_are_sane() -> None:
    column = ColumnInfo(name="x", type="TEXT")
    assert column.nullable is True
    assert column.primary_key is False


# --------------------------------------------------------------------------
# describe()
# --------------------------------------------------------------------------


def test_describe_returns_sample_rows(database: MockDatabase) -> None:
    description = database.describe("orders")
    assert 1 <= len(description.sample_rows) <= 3
    assert set(description.sample_rows[0]) == {c.name for c in description.table.columns}


def test_describe_reports_low_cardinality_values(database: MockDatabase) -> None:
    description = database.describe("orders")
    domains = {values.column: values.values for values in description.distinct_values}
    assert "status" in domains
    assert set(domains["status"]) == {"completed", "cancelled", "pending"}
    assert "channel" in domains


def test_describe_skips_high_cardinality_columns(database: MockDatabase) -> None:
    description = database.describe("customers")
    domains = {values.column for values in description.distinct_values}
    assert "email" not in domains
    assert "name" not in domains
    assert "segment" in domains


def test_describe_honours_the_sample_limit(database: MockDatabase) -> None:
    assert len(database.describe("orders", sample_rows=1).sample_rows) == 1
    assert len(database.describe("orders", sample_rows=5).sample_rows) == 5


@pytest.mark.parametrize("count", [0, -1, 21])
def test_describe_rejects_out_of_range_sample_counts(
    database: MockDatabase, count: int
) -> None:
    with pytest.raises(ValueError):
        database.describe("orders", sample_rows=count)


def test_describe_rejects_unknown_tables(database: MockDatabase) -> None:
    with pytest.raises(KeyError):
        database.describe("revenue_by_month")


def test_describe_still_works_after_the_authorizer_is_installed(
    database: MockDatabase,
) -> None:
    """Introspection metadata is cached; only SELECTs run on the locked connection."""
    assert database.describe("customers").table.row_count == 400


# --------------------------------------------------------------------------
# Read-only enforcement at engine level
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO customers (id, name) VALUES (9999, 'x')",
        "UPDATE customers SET name = 'x' WHERE id = 1",
        "DELETE FROM orders",
        "DROP TABLE orders",
        "ALTER TABLE orders ADD COLUMN extra TEXT",
        "CREATE TABLE t (id INTEGER)",
        "PRAGMA table_info(orders)",
        "ATTACH DATABASE ':memory:' AS other",
    ],
)
def test_writes_and_ddl_are_denied_by_the_engine(
    database: MockDatabase, sql: str
) -> None:
    with pytest.raises(sqlite3.DatabaseError):
        database.connection.execute(sql)


def test_selects_are_allowed(database: MockDatabase) -> None:
    assert database.connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0] > 0


def test_explicit_transactions_are_denied(database: MockDatabase) -> None:
    with pytest.raises(sqlite3.DatabaseError):
        database.connection.execute("BEGIN")


def test_data_is_unchanged_after_failed_write_attempts(database: MockDatabase) -> None:
    before = database.connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    with pytest.raises(sqlite3.DatabaseError):
        database.connection.execute("DELETE FROM customers")
    after = database.connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    assert before == after


def test_builds_are_independent() -> None:
    first = MockDatabase.build(seed=1, months=6, reference_date=ANCHOR)
    second = MockDatabase.build(seed=1, months=24, reference_date=ANCHOR)
    assert first.schema().table("orders").row_count != second.schema().table("orders").row_count
