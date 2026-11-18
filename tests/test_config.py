"""Configuration is validated once, with every problem reported together (E-06)."""

from __future__ import annotations

from datetime import date

import pytest

from core.config import Settings
from core.types import ConfigError


def test_valid_environment_builds_settings(env: dict[str, str]) -> None:
    settings = Settings.from_env(env)
    assert settings.database_backend == "mock"
    assert settings.llm_model == "test-model"
    assert settings.max_rows == 500
    assert settings.context_budget == 4000
    assert settings.llm_configured is True


def test_model_call_limits_have_conservative_defaults(env: dict[str, str]) -> None:
    settings = Settings.from_env(env)
    assert settings.llm_max_tokens == 2000
    assert settings.llm_max_retries == 3
    assert settings.llm_json_mode is True


def test_negative_retries_are_rejected(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError) as caught:
        Settings.from_env({**env, "LLM_MAX_RETRIES": "-1"})
    assert "LLM_MAX_RETRIES" in str(caught.value)


def test_missing_llm_credentials_are_reported_together() -> None:
    with pytest.raises(ConfigError) as caught:
        Settings.from_env({})
    problems = caught.value.problems
    assert any("LLM_API_KEY" in problem for problem in problems)
    assert any("LLM_MODEL" in problem for problem in problems)
    assert len(problems) == 2


def test_require_llm_false_allows_missing_credentials() -> None:
    settings = Settings.from_env({}, require_llm=False)
    assert settings.llm_configured is False


def test_invalid_integer_is_reported_with_its_variable_name(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError) as caught:
        Settings.from_env({**env, "MAX_ROWS": "lots"})
    assert "MAX_ROWS" in str(caught.value)


def test_all_problems_are_reported_in_one_raise(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError) as caught:
        Settings.from_env(
            {
                "MAX_ROWS": "x",
                "MAX_QUERIES": "y",
                "DATABASE_BACKEND": "mysql",
                "REFERENCE_DATE": "not-a-date",
            }
        )
    assert len(caught.value.problems) >= 4


def test_unknown_backend_is_rejected(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError) as caught:
        Settings.from_env({**env, "DATABASE_BACKEND": "duckdb"})
    assert "DATABASE_BACKEND" in str(caught.value)


def test_postgres_requires_database_url(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError) as caught:
        Settings.from_env({**env, "DATABASE_BACKEND": "postgres"})
    assert "DATABASE_URL" in str(caught.value)


def test_postgres_with_url_is_accepted(env: dict[str, str]) -> None:
    settings = Settings.from_env(
        {
            **env,
            "DATABASE_BACKEND": "postgres",
            "DATABASE_URL": "postgresql+psycopg://analyst:pw@localhost:5432/analytics",
        }
    )
    assert settings.database_backend == "postgres"
    assert settings.database_url is not None


def test_mock_backend_ignores_a_leftover_database_url(env: dict[str, str]) -> None:
    settings = Settings.from_env(
        {**env, "DATABASE_URL": "postgresql+psycopg://analyst:pw@localhost:5432/analytics"}
    )
    assert settings.database_backend == "mock"


def test_reference_date_parses_iso(env: dict[str, str]) -> None:
    settings = Settings.from_env({**env, "REFERENCE_DATE": "2026-03-15"})
    assert settings.reference_date == date(2026, 3, 15)


def test_reference_date_defaults_to_today(env: dict[str, str]) -> None:
    assert Settings.from_env(env).reference_date == date.today()


@pytest.mark.parametrize("raw", ["1", "true", "YES", "on"])
def test_boolean_true_variants(env: dict[str, str], raw: str) -> None:
    settings = Settings.from_env({**env, "LLM_SUPPORTS_TOOL_CALLING": raw})
    assert settings.llm_supports_tool_calling is True


@pytest.mark.parametrize("raw", ["0", "false", "No", "off"])
def test_boolean_false_variants(env: dict[str, str], raw: str) -> None:
    settings = Settings.from_env({**env, "LLM_SUPPORTS_TOOL_CALLING": raw})
    assert settings.llm_supports_tool_calling is False


def test_invalid_boolean_is_rejected(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        Settings.from_env({**env, "LLM_SUPPORTS_TOOL_CALLING": "maybe"})


def test_limits_below_minimum_are_rejected(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError) as caught:
        Settings.from_env({**env, "MAX_TOOL_TURNS": "0"})
    assert "MAX_TOOL_TURNS" in str(caught.value)


def test_public_config_hides_the_api_key(settings: Settings) -> None:
    public = settings.public_config()
    assert "sk-test-000000000000" not in str(public)
    assert public["llm_configured"] is True


def test_public_config_masks_the_dsn_password(env: dict[str, str]) -> None:
    settings = Settings.from_env(
        {
            **env,
            "DATABASE_BACKEND": "postgres",
            "DATABASE_URL": "postgresql+psycopg://analyst:supersecret@localhost:5432/analytics",
        }
    )
    masked = settings.public_config()["database_url"]
    assert masked is not None
    assert "supersecret" not in str(masked)
    assert "analyst" in str(masked)


def test_secrets_returns_values_to_redact(env: dict[str, str]) -> None:
    settings = Settings.from_env(env)
    assert settings.llm_api_key in settings.secrets()
