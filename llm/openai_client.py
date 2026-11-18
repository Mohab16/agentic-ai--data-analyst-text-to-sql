"""OpenAI-compatible chat completions: the only provider-specific module.

Two rules shape this file.

**We own the retry loop.** The SDK's own retry is disabled (`max_retries=0`) so
the number of attempts is knowable: it is printed, traced, and bounded by
`LLM_MAX_RETRIES`. Retries happen only for failures that can succeed on a second
attempt — rate limits, timeouts, connection errors, and 5xx.

**Errors are values with stable kinds.** A 401 and a 429 need different reactions
(fix credentials vs. wait), so the kind is part of the contract, and messages are
redacted before they leave this module (G-05).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import openai

from core.config import Settings
from core.trace import Redactor
from llm.base import (
    LLMError,
    LLMRequest,
    LLMResponse,
    ToolCall,
    ToolSpec,
)

_MESSAGE_LIMIT = 400


def _build_client(settings: Settings) -> openai.OpenAI:
    return openai.OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_s,
        max_retries=0,
    )


class OpenAICompatibleClient:
    """Satisfies `LLMClient` against any endpoint that speaks chat completions."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: openai.OpenAI | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._client = client or _build_client(settings)
        self._sleep = sleep
        self._redactor = Redactor(settings.secrets())

    def complete(self, request: LLMRequest) -> LLMResponse:
        limit = self._settings.llm_max_retries
        for attempt in range(limit + 1):
            attempts = attempt + 1
            try:
                response = self._create(request)
            except openai.OpenAIError as exc:
                error = self._map(exc, attempts=attempts)
                if not error.retryable or attempt >= limit:
                    raise error from exc
                self._sleep(_backoff(attempt))
                continue
            return _parse(response, attempts=attempts)

        # The loop returns or raises; this line only satisfies the type checker.
        raise LLMError("unknown", "no attempt was made")

    def _create(self, request: LLMRequest) -> Any:
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": _wire_messages(request),
            "max_tokens": request.max_tokens or self._settings.llm_max_tokens,
        }
        if request.json_mode and self._settings.llm_json_mode:
            payload["response_format"] = {"type": "json_object"}
        if request.tools:
            payload["tools"] = [_wire_tool(tool) for tool in request.tools]
        return self._client.chat.completions.create(**payload)

    def _map(self, exc: openai.OpenAIError, *, attempts: int) -> LLMError:
        kind, retryable = _classify(exc)
        status = getattr(exc, "status_code", None)
        message = " ".join(str(exc).split())[:_MESSAGE_LIMIT] or type(exc).__name__
        return LLMError(
            kind,
            self._redactor.redact(message),
            retryable=retryable,
            status=status if isinstance(status, int) else None,
            attempts=attempts,
        )


def _classify(exc: openai.OpenAIError) -> tuple[str, bool]:
    """Order matters: the SDK's exceptions form a hierarchy, not a flat enum."""
    if isinstance(exc, openai.APITimeoutError):
        return "timeout", True
    if isinstance(exc, openai.RateLimitError):
        return "rate_limit", True
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return "auth", False
    if isinstance(exc, openai.NotFoundError):
        return "not_found", False
    if isinstance(exc, (openai.BadRequestError, openai.UnprocessableEntityError)):
        return "bad_request", False
    if isinstance(exc, openai.InternalServerError):
        return "server", True
    if isinstance(exc, openai.APIConnectionError):
        return "connection", True
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", 0) or 0
        return ("server", True) if status >= 500 else ("api_error", False)
    return "unknown", False


def _backoff(attempt: int) -> float:
    return min(8.0, 0.5 * (2**attempt))


def _parse(response: Any, *, attempts: int) -> LLMResponse:
    choices = getattr(response, "choices", None) or ()
    if not choices:
        raise LLMError(
            "empty_response",
            "the provider returned no choices",
            attempts=attempts,
        )

    choice = choices[0]
    message = choice.message
    usage = getattr(response, "usage", None)

    return LLMResponse(
        text=message.content or "",
        model=getattr(response, "model", "") or "",
        tool_calls=tuple(
            _parse_tool_call(call) for call in (getattr(message, "tool_calls", None) or ())
        ),
        finish_reason=choice.finish_reason or "",
        tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
        tokens_out=getattr(usage, "completion_tokens", 0) or 0,
        attempts=attempts,
    )


def _parse_tool_call(call: Any) -> ToolCall:
    function = call.function
    raw = function.arguments or ""
    arguments: dict[str, Any] = {}
    error = ""
    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        error = f"arguments are not valid JSON: {exc.msg}"
    else:
        if isinstance(parsed, dict):
            arguments = parsed
        else:
            error = "arguments are not a JSON object"

    return ToolCall(
        id=call.id or "",
        name=function.name or "",
        arguments=arguments,
        raw_arguments=raw,
        parse_error=error,
    )


def _wire_messages(request: LLMRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": request.system}
    ]
    for message in request.messages:
        wire: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.role == "tool":
            wire["tool_call_id"] = message.tool_call_id or ""
            if message.name:
                wire["name"] = message.name
        if message.tool_calls:
            # A tool-requesting turn has no prose; providers reject an empty string.
            wire["content"] = message.content or None
            wire["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.raw_arguments
                        or json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
        messages.append(wire)
    return messages


def _wire_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }
