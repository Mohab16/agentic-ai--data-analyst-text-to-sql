"""Backend selection. One place chooses a concrete adapter (G-01, L-03)."""

from __future__ import annotations

from core.config import Settings
from core.types import BackendUnavailable
from db.base import Database


def build_database(settings: Settings) -> Database:
    """Construct the configured backend. Never connects when the mock is selected."""
    if settings.database_backend == "mock":
        from db.mock_source import MockDatabase

        return MockDatabase.build(
            seed=settings.mock_seed,
            months=settings.mock_months,
            reference_date=settings.reference_date,
        )
    if settings.database_backend == "postgres":
        from db.postgres_source import PostgresDatabase

        if not settings.database_url:
            raise BackendUnavailable(
                "DATABASE_URL is required for the PostgreSQL backend"
            )
        try:
            return PostgresDatabase.build(
                url=settings.database_url,
                reference_date=settings.reference_date,
            )
        except Exception as exc:
            raise BackendUnavailable(
                f"could not connect to PostgreSQL: {exc}"
            ) from exc
    raise BackendUnavailable(
        f"backend {settings.database_backend!r} is not recognized: "
        "use 'mock' or 'postgres'"
    )
