"""The mock dataset is deterministic and tells a findable story.

Assertions are made through SQL against a locally created connection, so they
test what an analyst would actually query, not the generator's internals.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from statistics import median

import pytest

from db.seed import (
    SeedConfig,
    build_dataset,
    create_schema,
    data_span,
    insert_dataset,
)

ANCHOR = date(2026, 11, 17)


def load(seed: int = 42, months: int = 24, anchor: date = ANCHOR) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    create_schema(connection)
    insert_dataset(
        connection, build_dataset(SeedConfig(seed=seed, months=months, reference_date=anchor))
    )
    return connection


def scalar(connection: sqlite3.Connection, sql: str, *params: object) -> float:
    return connection.execute(sql, params).fetchone()[0]


def monthly_revenue(connection: sqlite3.Connection) -> list[tuple[str, float]]:
    return connection.execute(
        """
        SELECT substr(o.order_date, 1, 7) AS month, ROUND(SUM(i.line_total), 2)
        FROM orders o JOIN order_items i ON i.order_id = o.id
        WHERE o.status = 'completed'
        GROUP BY month ORDER BY month
        """
    ).fetchall()


# --------------------------------------------------------------------------
# Determinism and shape
# --------------------------------------------------------------------------


def test_same_config_produces_identical_data() -> None:
    first = build_dataset(SeedConfig(seed=7, months=12, reference_date=ANCHOR))
    second = build_dataset(SeedConfig(seed=7, months=12, reference_date=ANCHOR))
    assert first == second


def test_different_seed_produces_different_data() -> None:
    first = build_dataset(SeedConfig(seed=7, months=12, reference_date=ANCHOR))
    second = build_dataset(SeedConfig(seed=8, months=12, reference_date=ANCHOR))
    assert first.orders != second.orders


def test_row_counts_are_sufficient_for_investigation() -> None:
    dataset = build_dataset(SeedConfig(seed=42, months=24, reference_date=ANCHOR))
    counts = dataset.counts()
    assert counts["customers"] == 400
    assert counts["products"] == 60
    assert counts["orders"] > 3_000
    assert counts["order_items"] > counts["orders"]
    assert counts["payments"] == counts["orders"]
    assert counts["returns"] > 100


def test_every_order_has_at_least_one_item() -> None:
    connection = load()
    orphans = scalar(
        connection,
        "SELECT COUNT(*) FROM orders o WHERE NOT EXISTS "
        "(SELECT 1 FROM order_items i WHERE i.order_id = o.id)",
    )
    assert orphans == 0


def test_every_foreign_key_resolves() -> None:
    connection = load(seed=3, months=12)
    checks = {
        "order_items.order_id": "SELECT COUNT(*) FROM order_items i WHERE NOT EXISTS "
        "(SELECT 1 FROM orders o WHERE o.id = i.order_id)",
        "order_items.product_id": "SELECT COUNT(*) FROM order_items i WHERE NOT EXISTS "
        "(SELECT 1 FROM products p WHERE p.id = i.product_id)",
        "orders.customer_id": "SELECT COUNT(*) FROM orders o WHERE NOT EXISTS "
        "(SELECT 1 FROM customers c WHERE c.id = o.customer_id)",
        "payments.order_id": "SELECT COUNT(*) FROM payments p WHERE NOT EXISTS "
        "(SELECT 1 FROM orders o WHERE o.id = p.order_id)",
        "returns.order_item_id": "SELECT COUNT(*) FROM returns r WHERE NOT EXISTS "
        "(SELECT 1 FROM order_items i WHERE i.id = r.order_item_id)",
    }
    for label, sql in checks.items():
        assert scalar(connection, sql) == 0, label


def test_refunds_never_exceed_the_line_total() -> None:
    connection = load()
    over = scalar(
        connection,
        "SELECT COUNT(*) FROM returns r JOIN order_items i ON i.id = r.order_item_id "
        "WHERE r.refund_amount > i.line_total + 0.01",
    )
    assert over == 0


def test_data_ends_the_day_before_the_reference_date() -> None:
    connection = load(anchor=ANCHOR)
    assert scalar(connection, "SELECT MAX(order_date) FROM orders") == "2026-11-16"
    assert scalar(connection, "SELECT COUNT(*) FROM orders WHERE order_date >= ?", "2026-11-17") == 0


def test_data_starts_at_the_configured_month() -> None:
    connection = load(months=24, anchor=ANCHOR)
    start, _ = data_span(SeedConfig(months=24, reference_date=ANCHOR))
    assert start == date(2024, 12, 1)
    assert scalar(connection, "SELECT MIN(order_date) FROM orders") == "2024-12-01"


def test_month_count_controls_the_history_length() -> None:
    short = load(months=6, anchor=ANCHOR)
    long = load(months=24, anchor=ANCHOR)
    assert scalar(short, "SELECT COUNT(DISTINCT substr(order_date,1,7)) FROM orders") == 6
    assert scalar(long, "SELECT COUNT(DISTINCT substr(order_date,1,7)) FROM orders") == 24


def test_current_month_is_partial() -> None:
    connection = load(anchor=ANCHOR)
    days_this_month = scalar(
        connection,
        "SELECT COUNT(DISTINCT order_date) FROM orders WHERE order_date >= '2026-11-01'",
    )
    assert 0 < days_this_month < 17


# --------------------------------------------------------------------------
# The planted story: revenue falls, North Electronics is gone, returns spike
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def seeded() -> sqlite3.Connection:
    return load(seed=42, months=24)


def test_revenue_falls_in_the_last_complete_month(seeded: sqlite3.Connection) -> None:
    months = monthly_revenue(seeded)
    previous, latest = months[-3], months[-2]  # last row is the partial month
    assert latest[0] == "2026-10" and previous[0] == "2026-09"
    assert latest[1] < previous[1] * 0.95


def test_north_electronics_goes_to_zero_while_it_was_selling_before(
    seeded: sqlite3.Connection,
) -> None:
    def north_electronics(month: str) -> float:
        return scalar(
            seeded,
            """
            SELECT COALESCE(ROUND(SUM(i.line_total), 2), 0)
            FROM orders o
            JOIN order_items i ON i.order_id = o.id
            JOIN products p ON p.id = i.product_id
            WHERE o.status = 'completed' AND o.shipping_region = 'North'
              AND p.category = 'Electronics' AND substr(o.order_date, 1, 7) = ?
            """,
            month,
        )

    assert north_electronics("2026-09") > 0
    assert north_electronics("2026-10") == 0


def test_electronics_share_collapses_in_the_drop_month(seeded: sqlite3.Connection) -> None:
    shares = seeded.execute(
        """
        SELECT substr(o.order_date, 1, 7) AS month,
               ROUND(100.0 * SUM(CASE WHEN p.category = 'Electronics' THEN i.line_total ELSE 0 END)
                     / SUM(i.line_total), 1)
        FROM orders o JOIN order_items i ON i.order_id = o.id JOIN products p ON p.id = i.product_id
        WHERE o.status = 'completed' AND substr(o.order_date, 1, 7) IN ('2026-09', '2026-10')
        GROUP BY month ORDER BY month
        """
    ).fetchall()
    previous, latest = shares
    assert latest[1] < previous[1] * 0.8


def test_fashion_returns_spike_in_the_drop_month(seeded: sqlite3.Connection) -> None:
    rates = seeded.execute(
        """
        SELECT substr(o.order_date, 1, 7) AS month,
               ROUND(1.0 * COUNT(r.id) / COUNT(i.id), 3)
        FROM orders o
        JOIN order_items i ON i.order_id = o.id
        JOIN products p ON p.id = i.product_id
        LEFT JOIN returns r ON r.order_item_id = i.id
        WHERE p.category = 'Fashion'
          AND substr(o.order_date, 1, 7) IN ('2026-09', '2026-10')
        GROUP BY month ORDER BY month
        """
    ).fetchall()
    previous, latest = rates
    assert latest[1] > previous[1] * 2


def test_the_drop_is_visible_for_several_seeds() -> None:
    """The story must not depend on one lucky seed."""
    for seed in (1, 2, 3, 4, 5):
        connection = load(seed=seed)
        months = monthly_revenue(connection)
        assert months[-2][1] < months[-3][1], f"seed {seed} shows no drop"
        connection.close()


def test_electronics_revenue_is_far_below_its_own_trend(seeded: sqlite3.Connection) -> None:
    """The drop is in one category, not a slow slide everywhere."""
    rows = seeded.execute(
        """
        SELECT substr(o.order_date, 1, 7) AS month, ROUND(SUM(i.line_total), 2)
        FROM orders o JOIN order_items i ON i.order_id = o.id JOIN products p ON p.id = i.product_id
        WHERE o.status = 'completed' AND p.category = 'Electronics'
        GROUP BY month ORDER BY month
        """
    ).fetchall()
    complete = rows[:-1]
    baseline = median(value for _, value in complete[:-1])
    assert complete[-1][1] < baseline * 0.75
