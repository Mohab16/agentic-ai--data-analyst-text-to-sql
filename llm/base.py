"""The model boundary: one request shape, one response shape, typed failures.

Everything that talks to a model imports this module. Only
`llm/openai_client.py` knows a provider wire format, so agents can be tested
against a twenty-line stand-in instead of a network.

The shapes are deliberately smaller than any provider's API: text, tool calls,
usage, and a small set of stable failure kinds. A provider that cannot supply
something leaves it empty rather than inventing a field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    """A call the model asked for. `arguments` is empty when the JSON was broken.

    `parse_error` is set in that case, so the caller can report a malformed call
    instead of silently invoking a tool with no arguments.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""
    parse_error: str = ""


@dataclass(frozen=True)
class Message:
    """One turn of a conversation.

    `system` is deliberately absent: the static instruction travels in
    `LLMRequest.system`, so it can never be confused with assembled context
    (X-06). An assistant turn that requests tools carries `tool_calls`; the
    matching results come back as `role="tool"` messages with `tool_call_id`.
    """

    role: Role
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class ToolSpec:
    """What the model is told it may call. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


@dataclass(frozen=True)
class LLMRequest:
    """One completion request. `json_mode` asks the provider for a JSON object."""

    system: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    json_mode: bool = False
    max_tokens: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    """One completion. `attempts` counts transport attempts, retries included."""

    text: str = ""
    model: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    attempts: int = 1


class LLMError(RuntimeError):
    """A model call failed. `kind` is stable; `message` is safe to print.

    Kinds: auth, rate_limit, timeout, connection, bad_request, not_found,
    server, empty_response, unknown. `retryable` says whether sending the same
    request again can plausibly succeed.
    """

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        retryable: bool = False,
        status: int | None = None,
        attempts: int = 1,
    ) -> None:
        self.kind = kind
        self.message = message
        self.retryable = retryable
        self.status = status
        self.attempts = attempts
        super().__init__(f"{kind}: {message}")


class LLMClient(Protocol):
    """What the agents need. One method, so a double is trivial."""

    def complete(self, request: LLMRequest) -> LLMResponse: ...
