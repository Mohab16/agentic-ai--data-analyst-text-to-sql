"""Shared fixtures.

Every test gets a clean environment: this module clears all known variables
before each test, so a developer's shell or `.env` cannot change results.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from core.config import ENV_KEYS, Settings
from llm.base import LLMError, LLMRequest, LLMResponse


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's .env must not influence these tests (G-02)."""
    monkeypatch.setattr("app.main.load_dotenv", lambda: None)


@pytest.fixture
def env() -> dict[str, str]:
    """A complete, valid configuration mapping. No process env involved."""
    return {
        "LLM_API_KEY": "sk-test-000000000000",
        "LLM_MODEL": "test-model",
        "DATABASE_BACKEND": "mock",
        "MOCK_SEED": "7",
        "MOCK_MONTHS": "12",
        "CONTEXT_BUDGET": "4000",
    }


@pytest.fixture
def settings(env: dict[str, str], tmp_path: Path) -> Settings:
    return Settings.from_env({**env, "TRACE_DIR": str(tmp_path / "traces")})


@pytest.fixture
def trace_dir(tmp_path: Path) -> Path:
    return tmp_path / "traces"


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch, env: dict[str, str], trace_dir: Path) -> None:
    """A fully configured environment: every variable set, no .env involved."""
    for name, value in {**env, "TRACE_DIR": str(trace_dir)}.items():
        monkeypatch.setenv(name, value)


class ScriptedLLM:
    """A stand-in `LLMClient`: replies in order, records every request.

    Replies are strings (wrapped as a response with token usage), `LLMResponse`
    objects, or `LLMError` instances (raised). Running out of replies is a test
    bug, so it fails loudly instead of returning something plausible.
    """

    def __init__(self, replies: Iterable[object]) -> None:
        self.replies = list(replies)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self.replies:
            raise AssertionError("ScriptedLLM has no replies left")
        reply = self.replies.pop(0)
        if isinstance(reply, LLMError):
            raise reply
        if isinstance(reply, str):
            return LLMResponse(
                text=reply, model="scripted", tokens_in=10, tokens_out=5
            )
        return reply  # type: ignore[return-value]


@pytest.fixture
def scripted_llm() -> type[ScriptedLLM]:
    """The class itself: each test builds its own with `scripted_llm([...])`."""
    return ScriptedLLM
