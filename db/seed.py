"""Deterministic mock warehouse: schema, seed data, and load helpers.

This module is pure data generation. It owns no connection lifecycle and no
introspection; `db/mock_source.py` builds on top of it, and tests can load the
same data into their own SQLite connection.

The dataset tells a specific, reproducible story so that investigation can be
evaluated: revenue falls in the most recent complete month, driven by an
Electronics stock-out in the North region, with a separate spike in Fashion
returns. The last month is partial (data ends yesterday).
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE customers (
        id           INTEGER PRIMARY KEY,
        name         TEXT    NOT NULL,
        email        TEXT    NOT NULL UNIQUE,
        segment      TEXT    NOT NULL CHECK (segment IN ('consumer', 'smb', 'enterprise')),
        city         TEXT    NOT NULL,
        region       TEXT    NOT NULL,
        signup_date  TEXT    NOT NULL,
        is_active    INTEGER NOT NULL CHECK (is_active IN (0, 1))
    )
    """,
    """
    CREATE TABLE products (
        id           INTEGER PRIMARY KEY,
        name         TEXT    NOT NULL,
        category     TEXT    NOT NULL,
        subcategory  TEXT    NOT NULL,
        unit_price   REAL    NOT NULL CHECK (unit_price > 0),
        cost         REAL    NOT NULL CHECK (cost > 0),
        launch_date  TEXT    NOT NULL,
        is_active    INTEGER NOT NULL CHECK (is_active IN (0, 1))
    )
    """,
    """
    CREATE TABLE orders (
        id               INTEGER PRIMARY KEY,
        customer_id      INTEGER NOT NULL REFERENCES customers (id),
        order_date       TEXT    NOT NULL,
        status           TEXT    NOT NULL CHECK (status IN ('completed', 'cancelled', 'pending')),
        channel          TEXT    NOT NULL,
        shipping_city    TEXT    NOT NULL,
        shipping_region  TEXT    NOT NULL,
        discount_total   REAL    NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE order_items (
        id               INTEGER PRIMARY KEY,
        order_id         INTEGER NOT NULL REFERENCES orders (id),
        product_id       INTEGER NOT NULL REFERENCES products (id),
        quantity         INTEGER NOT NULL CHECK (quantity > 0),
        unit_price       REAL    NOT NULL CHECK (unit_price > 0),
        discount_amount  REAL    NOT NULL DEFAULT 0,
        line_total       REAL    NOT NULL CHECK (line_total >= 0)
    )
    """,
    """
    CREATE TABLE payments (
        id          INTEGER PRIMARY KEY,
        order_id    INTEGER NOT NULL REFERENCES orders (id),
        paid_at     TEXT,
        method      TEXT    NOT NULL,
        amount      REAL    NOT NULL,
        status      TEXT    NOT NULL CHECK (status IN ('captured', 'failed', 'pending'))
    )
    """,
    """
    CREATE TABLE returns (
        id             INTEGER PRIMARY KEY,
        order_item_id  INTEGER NOT NULL REFERENCES order_items (id),
        returned_at    TEXT    NOT NULL,
        quantity       INTEGER NOT NULL CHECK (quantity > 0),
        reason         TEXT    NOT NULL,
        refund_amount  REAL    NOT NULL CHECK (refund_amount >= 0)
    )
    """,
)

INDEX_STATEMENTS: tuple[str, ...] = (
    "CREATE INDEX idx_orders_customer ON orders (customer_id)",
    "CREATE INDEX idx_orders_date ON orders (order_date)",
    "CREATE INDEX idx_order_items_order ON order_items (order_id)",
    "CREATE INDEX idx_order_items_product ON order_items (product_id)",
    "CREATE INDEX idx_payments_order ON payments (order_id)",
    "CREATE INDEX idx_returns_order_item ON returns (order_item_id)",
    "CREATE INDEX idx_returns_date ON returns (returned_at)",
)

FIRST_NAMES = (
    "Amara", "Ben", "Chen", "Dara", "Eli", "Farah", "Gus", "Hana", "Ivan", "Jo",
    "Kofi", "Lena", "Milo", "Nadia", "Omar", "Petra", "Quinn", "Rosa", "Samir",
    "Tara", "Umar", "Vera", "Wes", "Yara",
)
LAST_NAMES = (
    "Adams", "Bello", "Costa", "Duval", "Eriksen", "Fischer", "Gul", "Haddad",
    "Ibrahim", "Jensen", "Kaur", "Lindqvist", "Moreau", "Novak", "Okafor",
    "Petrov", "Quintero", "Rossi", "Silva", "Tanaka", "Ueda", "Vargas", "Weber", "Zhang",
)
ADJECTIVES = (
    "Aurora", "Basalt", "Cedar", "Delta", "Ember", "Fjord", "Granite", "Harbor",
    "Indigo", "Juniper", "Kestrel", "Lumen", "Marlow", "Nordic", "Onyx", "Prairie",
    "Quartz", "Ridge", "Sable", "Tundra", "Umber", "Vantage", "Willow", "Zephyr",
)
NOUNS = (
    "Wave", "Peak", "Core", "Nova", "Loop", "Arc", "Halo", "Vista", "Drift",
    "Forge", "Pulse", "Atlas", "Crest", "Echo", "Orbit", "Slate", "Trace", "Spark",
)

REGIONS: dict[str, tuple[str, ...]] = {
    "North": ("Manchester", "Leeds", "Sheffield"),
    "South": ("London", "Brighton", "Southampton"),
    "East": ("Norwich", "Cambridge", "Ipswich"),
    "West": ("Bristol", "Cardiff", "Exeter"),
}

CATEGORIES: dict[str, tuple[tuple[str, ...], tuple[float, float]]] = {
    "Electronics": (("Audio", "Laptops", "Phones", "Accessories"), (150.0, 450.0)),
    "Home": (("Kitchen", "Furniture", "Decor"), (30.0, 260.0)),
    "Fashion": (("Outerwear", "Footwear", "Bags"), (20.0, 140.0)),
    "Sports": (("Fitness", "Outdoor", "Team"), (20.0, 220.0)),
    "Beauty": (("Skincare", "Haircare", "Fragrance"), (10.0, 70.0)),
}

PRODUCTS_PER_CATEGORY = 12
CUSTOMER_COUNT = 400

SEGMENTS = (("consumer", 62), ("smb", 28), ("enterprise", 10))
CHANNELS = (("web", 46), ("mobile", 38), ("marketplace", 16))
ITEM_COUNTS = ((1, 44), (2, 31), (3, 17), (4, 8))
QUANTITIES = ((1, 74), (2, 20), (3, 6))
RETURN_REASONS = (
    ("damaged", 30),
    ("wrong_item", 20),
    ("changed_mind", 24),
    ("late_delivery", 16),
    ("not_as_described", 10),
)
PAYMENT_METHODS = (
    ("card", 52),
    ("wallet", 24),
    ("bank_transfer", 16),
    ("cash_on_delivery", 8),
)

# The planted story, as parameters. Skipping is deterministic (every Nth item),
# so the effect is stable across seeds instead of being one more noise source.
# North Electronics is fully unavailable for one month; elsewhere half is.
NORTH_STOCKOUT_EVERY = 1
OTHER_STOCKOUT_EVERY = 2
BASELINE_RETURN_RATE = 0.06
FASHION_RETURN_SPIKE = 0.30


@dataclass(frozen=True)
class SeedConfig:
    seed: int = 42
    months: int = 24
    reference_date: date = date.today()


@dataclass(frozen=True)
class Dataset:
    customers: tuple[dict[str, Any], ...]
    products: tuple[dict[str, Any], ...]
    orders: tuple[dict[str, Any], ...]
    order_items: tuple[dict[str, Any], ...]
    payments: tuple[dict[str, Any], ...]
    returns: tuple[dict[str, Any], ...]

    def counts(self) -> dict[str, int]:
        return {
            "customers": len(self.customers),
            "products": len(self.products),
            "orders": len(self.orders),
            "order_items": len(self.order_items),
            "payments": len(self.payments),
            "returns": len(self.returns),
        }


def create_schema(connection: sqlite3.Connection) -> None:
    """Create tables and indexes. Explicit, never called at import time (G-04)."""
    for statement in SCHEMA_STATEMENTS:
        connection.execute(statement)
    for statement in INDEX_STATEMENTS:
        connection.execute(statement)


def insert_dataset(connection: sqlite3.Connection, dataset: Dataset) -> None:
    """Insert the whole dataset in one transaction."""
    rows_by_table = (
        ("customers", dataset.customers),
        ("products", dataset.products),
        ("orders", dataset.orders),
        ("order_items", dataset.order_items),
        ("payments", dataset.payments),
        ("returns", dataset.returns),
    )
    with connection:
        for table, rows in rows_by_table:
            if not rows:
                continue
            columns = list(rows[0].keys())
            placeholders = ", ".join("?" for _ in columns)
            sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
            connection.executemany(sql, [tuple(row[c] for c in columns) for row in rows])


def build_dataset(config: SeedConfig) -> Dataset:
    """Generate the dataset. Same config means byte-identical output."""
    rng = random.Random(config.seed)
    anchor = config.reference_date
    end = anchor - timedelta(days=1)
    start = _month_start(_add_months(anchor, -(config.months - 1)))
    drop_start = _month_start(_add_months(anchor, -1))
    drop_end = _month_start(anchor) - timedelta(days=1)

    customers = _build_customers(rng, start, end)
    products, weights = _build_products(rng, start)
    orders, order_items, payments, returns = _build_activity(
        rng,
        customers,
        products,
        weights,
        start=start,
        end=end,
        drop_start=drop_start,
        drop_end=drop_end,
        anchor=anchor,
    )

    return Dataset(
        customers=tuple(customers),
        products=tuple(products),
        orders=tuple(orders),
        order_items=tuple(order_items),
        payments=tuple(payments),
        returns=tuple(returns),
    )


def data_span(config: SeedConfig) -> tuple[date, date]:
    """First and last day that contain data, for the schema freshness note."""
    return (
        _month_start(_add_months(config.reference_date, -(config.months - 1))),
        config.reference_date - timedelta(days=1),
    )


# --------------------------------------------------------------------------
# Entity generation
# --------------------------------------------------------------------------


def _build_customers(
    rng: random.Random, start: date, end: date
) -> list[dict[str, Any]]:
    span = max(1, (end - start).days)
    region_names = tuple(REGIONS)
    customers: list[dict[str, Any]] = []
    for index in range(CUSTOMER_COUNT):
        region = region_names[index % len(region_names)]
        signup = start + timedelta(days=rng.randint(0, span))
        customers.append(
            {
                "id": index + 1,
                "name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
                "email": f"customer{index + 1:04d}@example.com",
                "segment": _weighted(rng, SEGMENTS),
                "city": rng.choice(REGIONS[region]),
                "region": region,
                "signup_date": signup.isoformat(),
                "is_active": 0 if rng.random() < 0.08 else 1,
            }
        )
    return customers


def _build_products(
    rng: random.Random, start: date
) -> tuple[list[dict[str, Any]], list[float]]:
    """Products plus their popularity weights (used for order composition)."""
    products: list[dict[str, Any]] = []
    weights: list[float] = []
    used_names: set[str] = set()
    for category, (subcategories, (low, high)) in CATEGORIES.items():
        for _ in range(PRODUCTS_PER_CATEGORY):
            subcategory = rng.choice(subcategories)
            name = _unique_name(rng, subcategory, used_names)
            price = round(rng.uniform(low, high), 2)
            launch = start + timedelta(days=rng.randint(0, 540))
            products.append(
                {
                    "id": len(products) + 1,
                    "name": name,
                    "category": category,
                    "subcategory": subcategory,
                    "unit_price": price,
                    "cost": round(price * rng.uniform(0.45, 0.72), 2),
                    "launch_date": launch.isoformat(),
                    "is_active": 0 if rng.random() < 0.07 else 1,
                }
            )
            weights.append(rng.uniform(0.8, 1.6))
    # The flagship product: the one that will be out of stock in the North.
    weights[0] = 2.5
    return products, weights


def _unique_name(rng: random.Random, subcategory: str, used: set[str]) -> str:
    for _ in range(64):
        name = f"{rng.choice(ADJECTIVES)} {rng.choice(NOUNS)} {subcategory}"
        if name not in used:
            used.add(name)
            return name
    serial = len(used) + 1
    name = f"{subcategory} Edition {serial:03d}"
    used.add(name)
    return name


def _build_activity(
    rng: random.Random,
    customers: list[dict[str, Any]],
    products: list[dict[str, Any]],
    weights: list[float],
    *,
    start: date,
    end: date,
    drop_start: date,
    drop_end: date,
    anchor: date,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    orders: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    payments: list[dict[str, Any]] = []
    returns: list[dict[str, Any]] = []
    region_names = tuple(REGIONS)
    electronics_seen: dict[str, int] = {region: 0 for region in region_names}

    for day_index in range((end - start).days + 1):
        day = start + timedelta(days=day_index)
        month_index = (day.year - start.year) * 12 + (day.month - start.month)
        in_drop_month = drop_start <= day <= drop_end
        for _ in range(_orders_on(rng, day, month_index)):
            customer = customers[rng.randrange(len(customers))]
            region = customer["region"]
            if rng.random() > 0.85:
                region = rng.choice(region_names)
            city = customer["city"] if region == customer["region"] else rng.choice(
                REGIONS[region]
            )
            status = _order_status(rng, day, end)

            order_id = len(orders) + 1
            order_total = 0.0
            item_rows: list[dict[str, Any]] = []

            for _ in range(_weighted(rng, ITEM_COUNTS)):
                product = rng.choices(products, weights=weights, k=1)[0]
                if in_drop_month and product["category"] == "Electronics":
                    electronics_seen[region] += 1
                    every = (
                        NORTH_STOCKOUT_EVERY
                        if region == "North"
                        else OTHER_STOCKOUT_EVERY
                    )
                    if electronics_seen[region] % every == 0:
                        continue
                quantity = _weighted(rng, QUANTITIES)
                unit_price = product["unit_price"]
                gross = quantity * unit_price
                discount = (
                    0.0
                    if rng.random() < 0.62
                    else round(gross * rng.choice((0.05, 0.10, 0.15, 0.20)), 2)
                )
                line_total = round(gross - discount, 2)
                order_total += line_total
                item_rows.append(
                    {
                        "id": len(items) + len(item_rows) + 1,
                        "order_id": order_id,
                        "product_id": product["id"],
                        "quantity": quantity,
                        "unit_price": unit_price,
                        "discount_amount": discount,
                        "line_total": line_total,
                        "_product": product,
                        "_ordered_on": day,
                    }
                )

            if not item_rows:
                # Never emit an order without items. During the stock-out the
                # replacement is deliberately not Electronics, so the regional
                # shortfall stays exact for the evaluator.
                choices = [
                    (product, weight)
                    for product, weight in zip(products, weights)
                    if not (in_drop_month and product["category"] == "Electronics")
                ]
                product = rng.choices(
                    [p for p, _ in choices], weights=[w for _, w in choices], k=1
                )[0]
                quantity = _weighted(rng, QUANTITIES)
                line_total = round(quantity * product["unit_price"], 2)
                order_total += line_total
                item_rows.append(
                    {
                        "id": len(items) + 1,
                        "order_id": order_id,
                        "product_id": product["id"],
                        "quantity": quantity,
                        "unit_price": product["unit_price"],
                        "discount_amount": 0.0,
                        "line_total": line_total,
                        "_product": product,
                        "_ordered_on": day,
                    }
                )

            orders.append(
                {
                    "id": order_id,
                    "customer_id": customer["id"],
                    "order_date": day.isoformat(),
                    "status": status,
                    "channel": _weighted(rng, CHANNELS),
                    "shipping_city": city,
                    "shipping_region": region,
                    "discount_total": round(
                        sum(row["discount_amount"] for row in item_rows), 2
                    ),
                }
            )

            payments.append(
                {
                    "id": len(payments) + 1,
                    "order_id": order_id,
                    "paid_at": day.isoformat() if status != "pending" else None,
                    "method": _weighted(rng, PAYMENT_METHODS),
                    "amount": round(order_total, 2) if status == "completed" else 0.0,
                    "status": {
                        "completed": "captured",
                        "cancelled": "failed",
                        "pending": "pending",
                    }[status],
                }
            )

            for row in item_rows:
                product = row.pop("_product")
                ordered_on = row.pop("_ordered_on")
                items.append(row)
                if status != "completed":
                    continue
                rate = BASELINE_RETURN_RATE
                if in_drop_month and product["category"] == "Fashion":
                    rate = FASHION_RETURN_SPIKE
                if rng.random() >= rate:
                    continue
                returned_at = ordered_on + timedelta(days=rng.randint(3, 21))
                if returned_at > anchor:
                    continue
                reasons = RETURN_REASONS
                if in_drop_month and product["category"] == "Fashion":
                    reasons = (("damaged", 70), ("wrong_item", 12), ("late_delivery", 10), ("changed_mind", 8))
                returns.append(
                    {
                        "id": len(returns) + 1,
                        "order_item_id": row["id"],
                        "returned_at": returned_at.isoformat(),
                        "quantity": row["quantity"],
                        "reason": _weighted(rng, reasons),
                        "refund_amount": round(
                            row["line_total"] * rng.choice((1.0, 1.0, 0.5)), 2
                        ),
                    }
                )

    return orders, items, payments, returns


def _orders_on(rng: random.Random, day: date, month_index: int) -> int:
    weekday_factor = (1.0, 1.0, 1.0, 1.05, 1.25, 0.8, 0.65)[day.weekday()]
    growth = 1.0 + 0.012 * month_index
    base = 8.0 * growth * weekday_factor
    return max(0, int(round(rng.gauss(base, base * 0.12))))


def _order_status(rng: random.Random, day: date, end: date) -> str:
    if (end - day).days <= 1 and rng.random() < 0.06:
        return "pending"
    return "cancelled" if rng.random() < 0.05 else "completed"


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _weighted(rng: random.Random, pairs: tuple[tuple[str, int], ...]) -> str:
    return rng.choices(
        [value for value, _ in pairs], weights=[weight for _, weight in pairs], k=1
    )[0]


def _month_start(day: date) -> date:
    return day.replace(day=1)


def _add_months(day: date, months: int) -> date:
    total = day.year * 12 + (day.month - 1) + months
    return date(total // 12, total % 12 + 1, 1)
