"""Schema rendering and the inspect_schema tool."""

from __future__ import annotations

from datetime import date

import pytest

from db.base import ColumnInfo, SchemaSnapshot, TableDescription, TableInfo
from db.mock_source import MockDatabase
from tools.schema import (
    make_inspect_schema_tool,
    render_schema_digest,
    render_table_detail,
)

ANCHOR = date(2026, 11, 17)


@pytest.fixture(scope="module")
def database() -> MockDatabase:
    return MockDatabase.build(seed=42, months=24, reference_date=ANCHOR)


@pytest.fixture(scope="module")
def inspect_schema(database: MockDatabase):
    return make_inspect_schema_tool(database)


# --------------------------------------------------------------------------
# Digest
# --------------------------------------------------------------------------


def test_digest_names_every_table(database: MockDatabase) -> None:
    text, truncated = render_schema_digest(database.schema(), 24_000)
    for name in database.schema().table_names:
        assert name in text
    assert truncated is False


def test_digest_states_the_backend_and_reference_date(database: MockDatabase) -> None:
    text, _ = render_schema_digest(database.schema(), 24_000)
    assert "backend: mock" in text
    assert "as of: 2026-11-17" in text


def test_digest_reports_the_partial_current_month(database: MockDatabase) -> None:
    text, _ = render_schema_digest(database.schema(), 24_000)
    assert "partial" in text
    assert "2026-11-16" in text


def test_digest_shows_types_keys_and_foreign_keys(database: MockDatabase) -> None:
    text, _ = render_schema_digest(database.schema(), 24_000)
    assert "INTEGER" in text
    assert "PK" in text
    assert "-> orders.id" in text


def test_digest_shows_row_counts(database: MockDatabase) -> None:
    text, _ = render_schema_digest(database.schema(), 24_000)
    assert "rows)" in text


def test_digest_points_at_the_detail_tool(database: MockDatabase) -> None:
    text, _ = render_schema_digest(database.schema(), 24_000)
    assert "inspect_schema" in text


def test_digest_respects_the_budget(database: MockDatabase) -> None:
    for budget in (2_000, 1_200, 800, 400, 200):
        text, _ = render_schema_digest(database.schema(), budget)
        assert len(text) <= budget


def test_digest_truncation_is_announced(database: MockDatabase) -> None:
    text, truncated = render_schema_digest(database.schema(), 900)
    assert truncated is True
    assert "truncated" in text


def test_truncated_digest_never_contains_a_half_table(database: MockDatabase) -> None:
    """Table blocks are atomic: no column line appears without its table header."""
    text, truncated = render_schema_digest(database.schema(), 900)
    assert truncated is True
    for block in text.split("\n\n")[1:]:  # block 0 is the header
        assert not block.startswith("  "), f"orphan block: {block[:40]!r}"


def test_small_budget_falls_back_to_compact_tables() -> None:
    columns = tuple(
        ColumnInfo(name=f"column_{index:02d}", type="TEXT") for index in range(20)
    )
    snapshot = SchemaSnapshot(
        backend="mock",
        as_of="2026-01-01",
        tables=(TableInfo(name="orders", columns=columns, row_count=10),),
    )
    full, _ = render_schema_digest(snapshot, 10_000)
    text, truncated = render_schema_digest(snapshot, len(full) - 40)
    assert "orders (10 rows): column_00" in text
    assert truncated is False


def test_zero_budget_returns_empty_text() -> None:
    text, truncated = render_schema_digest(
        SchemaSnapshot(backend="mock", as_of="2026-01-01", tables=()), 0
    )
    assert text == ""
    assert truncated is True


def test_digest_is_deterministic(database: MockDatabase) -> None:
    assert render_schema_digest(database.schema(), 4_000) == render_schema_digest(
        database.schema(), 4_000
    )


# --------------------------------------------------------------------------
# inspect_schema tool
# --------------------------------------------------------------------------


def test_inspect_schema_returns_columns_values_and_samples(inspect_schema) -> None:
    text = inspect_schema("orders")
    assert isinstance(text, str)
    assert "orders" in text
    assert "status" in text
    assert "completed" in text
    assert "sample rows" in text


def test_inspect_schema_marks_primary_and_foreign_keys(inspect_schema) -> None:
    text = inspect_schema("order_items")
    assert "PK" in text
    assert "-> orders.id" in text


def test_inspect_schema_lists_value_domains(inspect_schema) -> None:
    text = inspect_schema("products")
    assert "value domains" in text
    assert "Electronics" in text


def test_inspect_schema_rejects_unknown_table_with_known_names(inspect_schema) -> None:
    result = inspect_schema("revenue_by_month")
    assert not isinstance(result, str)
    assert result.kind == "invalid_input"
    assert result.tool == "inspect_schema"
    assert "orders" in result.detail


def test_inspect_schema_rejects_empty_table(inspect_schema) -> None:
    result = inspect_schema("")
    assert not isinstance(result, str)
    assert result.kind == "invalid_input"


@pytest.mark.parametrize(
    "hostile",
    [
        'orders"; DROP TABLE orders; --',
        "orders; DELETE FROM orders",
        "../../etc/passwd",
        "orders WHERE 1=1",
    ],
)
def test_inspect_schema_never_accepts_arbitrary_sql(
    inspect_schema, hostile: str
) -> None:
    result = inspect_schema(hostile)
    assert not isinstance(result, str)
    assert result.kind == "invalid_input"


def test_database_is_intact_after_hostile_table_names(
    inspect_schema, database: MockDatabase
) -> None:
    inspect_schema("orders; DROP TABLE orders")
    assert database.describe("orders").table.row_count > 0


def test_inspect_schema_accepts_quoted_names(inspect_schema) -> None:
    text = inspect_schema('"customers"')
    assert isinstance(text, str)
    assert "customers" in text


def test_render_table_detail_formats_empty_tables() -> None:
    empty = TableDescription(table=TableInfo(name="empty", columns=(), row_count=0))
    text = render_table_detail(empty)
    assert "table is empty" in text
    assert "no low-cardinality columns" in text
