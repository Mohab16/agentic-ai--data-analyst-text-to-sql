"""Configuration: one `Settings` object, constructed and validated once.

Rules: G-01 (no `os.getenv` outside this module), G-02 (`.env` loaded only in
the entry point), G-03 (limits are configuration), E-06 (fail fast, all
problems in one message).

`from_env` takes a mapping so tests never touch the process environment.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from core.types import Backend, ConfigError
from core.utils import mask_dsn

# Every variable this module reads. Tests clear these to isolate themselves.
ENV_KEYS: tuple[str, ...] = (
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_SUPPORTS_TOOL_CALLING",
    "LLM_TIMEOUT_S",
    "LLM_MAX_TOKENS",
    "LLM_MAX_RETRIES",
    "LLM_JSON_MODE",
    "DATABASE_BACKEND",
    "DATABASE_URL",
    "REFERENCE_DATE",
    "MOCK_SEED",
    "MOCK_MONTHS",
    "MAX_STEPS",
    "MAX_TOOL_TURNS",
    "MAX_QUERIES",
    "MAX_QUERY_RETRIES",
    "MAX_ROWS",
    "MAX_SQL_LENGTH",
    "STATEMENT_TIMEOUT_MS",
    "CONTEXT_BUDGET",
    "WALL_CLOCK_S",
    "TRACE_DIR",
    "TRACE_SAMPLE_ROWS",
)


@dataclass(frozen=True)
class Settings:
    # LLM (OpenAI-compatible). Empty when the LLM is not required yet.
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_supports_tool_calling: bool
    llm_timeout_s: int
    llm_max_tokens: int
    llm_max_retries: int
    llm_json_mode: bool

    # Database backend selection and mock data anchor
    database_backend: Backend
    database_url: str | None
    reference_date: date
    mock_seed: int
    mock_months: int

    # Investigation limits (O-03)
    max_steps: int
    max_tool_turns: int
    # Statements that may reach the database in one investigation.
    max_queries: int
    # Failures allowed after the first attempt (0 means no retry).
    max_query_retries: int
    max_rows: int
    max_sql_length: int
    statement_timeout_ms: int
    context_budget: int
    wall_clock_s: int

    # Observability (R-02, R-03)
    trace_dir: Path
    # Rows of each result kept as evidence in the report (0 keeps none).
    trace_sample_rows: int

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_model)

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None, *, require_llm: bool = True
    ) -> Settings:
        """Build and validate settings. Raises `ConfigError` listing every problem.

        `require_llm=False` is used by commands that cannot call a model
        (validation, schema inspection); it skips only the LLM credential check.
        """
        source: Mapping[str, str] = os.environ if env is None else env
        problems: list[str] = []

        api_key = _text(source, "LLM_API_KEY")
        model = _text(source, "LLM_MODEL")
        if require_llm:
            if not api_key:
                problems.append("LLM_API_KEY is required")
            if not model:
                problems.append("LLM_MODEL is required")

        backend = _text(source, "DATABASE_BACKEND", "mock").strip().lower()
        if backend not in ("mock", "postgres"):
            problems.append(
                f"DATABASE_BACKEND must be 'mock' or 'postgres', got {backend!r}"
            )
            backend = "mock"

        database_url = _text(source, "DATABASE_URL") or None
        if backend == "postgres" and not database_url:
            problems.append("DATABASE_URL is required when DATABASE_BACKEND=postgres")

        settings = cls(
            llm_api_key=api_key,
            llm_base_url=_text(source, "LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_model=model,
            llm_supports_tool_calling=_flag(
                source, "LLM_SUPPORTS_TOOL_CALLING", True, problems
            ),
            llm_timeout_s=_int(source, "LLM_TIMEOUT_S", 120, problems, minimum=1),
            llm_max_tokens=_int(
                source, "LLM_MAX_TOKENS", 2000, problems, minimum=1
            ),
            llm_max_retries=_int(source, "LLM_MAX_RETRIES", 3, problems, minimum=0),
            llm_json_mode=_flag(source, "LLM_JSON_MODE", True, problems),
            database_backend=backend,  # type: ignore[arg-type]
            database_url=database_url,
            reference_date=_date(source, "REFERENCE_DATE", problems),
            mock_seed=_int(source, "MOCK_SEED", 42, problems, minimum=0),
            mock_months=_int(source, "MOCK_MONTHS", 24, problems, minimum=3),
            max_steps=_int(source, "MAX_STEPS", 12, problems, minimum=1),
            max_tool_turns=_int(source, "MAX_TOOL_TURNS", 12, problems, minimum=1),
            max_queries=_int(source, "MAX_QUERIES", 8, problems, minimum=1),
            max_query_retries=_int(source, "MAX_QUERY_RETRIES", 2, problems, minimum=0),
            max_rows=_int(source, "MAX_ROWS", 500, problems, minimum=1),
            max_sql_length=_int(
                source, "MAX_SQL_LENGTH", 20_000, problems, minimum=200
            ),
            statement_timeout_ms=_int(
                source, "STATEMENT_TIMEOUT_MS", 15_000, problems, minimum=100
            ),
            context_budget=_int(source, "CONTEXT_BUDGET", 24_000, problems, minimum=500),
            wall_clock_s=_int(source, "WALL_CLOCK_S", 600, problems, minimum=5),
            trace_dir=Path(_text(source, "TRACE_DIR", "traces")),
            trace_sample_rows=_int(
                source, "TRACE_SAMPLE_ROWS", 5, problems, minimum=0
            ),
        )

        if problems:
            raise ConfigError(problems)
        return settings

    def public_config(self) -> dict[str, object]:
        """Settings safe to print or trace: no credentials, no full DSN."""
        return {
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "llm_supports_tool_calling": self.llm_supports_tool_calling,
            "llm_configured": self.llm_configured,
            "llm_timeout_s": self.llm_timeout_s,
            "llm_max_tokens": self.llm_max_tokens,
            "llm_max_retries": self.llm_max_retries,
            "llm_json_mode": self.llm_json_mode,
            "database_backend": self.database_backend,
            "database_url": _mask_dsn(self.database_url),
            "reference_date": self.reference_date.isoformat(),
            "mock_seed": self.mock_seed,
            "mock_months": self.mock_months,
            "max_steps": self.max_steps,
            "max_tool_turns": self.max_tool_turns,
            "max_queries": self.max_queries,
            "max_query_retries": self.max_query_retries,
            "max_rows": self.max_rows,
            "max_sql_length": self.max_sql_length,
            "statement_timeout_ms": self.statement_timeout_ms,
            "context_budget": self.context_budget,
            "wall_clock_s": self.wall_clock_s,
            "trace_dir": str(self.trace_dir),
            "trace_sample_rows": self.trace_sample_rows,
        }

    def secrets(self) -> tuple[str, ...]:
        """Values that must never appear in a trace or a user-facing message."""
        return tuple(v for v in (self.llm_api_key, self.database_url) if v)


def _text(source: Mapping[str, str], name: str, default: str = "") -> str:
    raw = source.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _int(
    source: Mapping[str, str],
    name: str,
    default: int,
    problems: list[str],
    *,
    minimum: int,
) -> int:
    raw = source.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        problems.append(f"{name} must be an integer, got {raw.strip()!r}")
        return default
    if value < minimum:
        problems.append(f"{name} must be >= {minimum}, got {value}")
        return default
    return value


def _flag(
    source: Mapping[str, str], name: str, default: bool, problems: list[str]
) -> bool:
    raw = source.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    problems.append(f"{name} must be a boolean, got {raw.strip()!r}")
    return default


def _date(source: Mapping[str, str], name: str, problems: list[str]) -> date:
    raw = source.get(name)
    if raw is None or not raw.strip():
        return date.today()
    try:
        return date.fromisoformat(raw.strip())
    except ValueError:
        problems.append(f"{name} must be an ISO date (YYYY-MM-DD), got {raw.strip()!r}")
        return date.today()


def _mask_dsn(dsn: str | None) -> str | None:
    """Keep scheme, user, and host visible; hide the password.
    
    Delegates to the centralized ``core.utils.mask_dsn`` to ensure all modules
    use the same implementation and future changes propagate automatically.
    """
    return mask_dsn(dsn)
