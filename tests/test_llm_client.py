"""The OpenAI-compatible client, tested through the real SDK without a network.

`httpx.MockTransport` sits at the bottom of the stack, so these tests exercise the
actual request body the SDK sends and the actual responses it parses. Nothing
here mocks `openai`; only the socket is replaced.
"""

from __future__ import annotations

import json

import httpx
import openai
import pytest

from core.config import Settings
from llm.base import LLMError, LLMRequest, Message, ToolCall, ToolSpec
from llm.openai_client import OpenAICompatibleClient

API_KEY = "sk-test-secret-value-123"


def make_settings(**overrides: str) -> Settings:
    return Settings.from_env(
        {
            "LLM_API_KEY": API_KEY,
            "LLM_MODEL": "test-model",
            "LLM_BASE_URL": "https://api.test/v1",
            **overrides,
        }
    )


def completion(
    text: str = "",
    *,
    tool_calls: list[dict[str, object]] | None = None,
    usage: tuple[int, int] | None = (12, 7),
    finish_reason: str = "stop",
) -> httpx.Response:
    message: dict[str, object] = {"role": "assistant", "content": text}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    body: dict[str, object] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if usage is not None:
        body["usage"] = {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": usage[0] + usage[1],
        }
    return httpx.Response(200, json=body)


class Transport:
    """Answers with queued responses (or raises queued exceptions) and records."""

    def __init__(self, replies: list[object]) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.replies:
            raise AssertionError("no queued reply")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        assert isinstance(reply, httpx.Response)
        return reply

    def body(self, index: int = 0) -> dict[str, object]:
        return json.loads(self.requests[index].content)

    @property
    def count(self) -> int:
        return len(self.requests)


def build(
    settings: Settings,
    transport: Transport,
    *,
    sleeps: list[float] | None = None,
) -> OpenAICompatibleClient:
    http = httpx.Client(transport=httpx.MockTransport(transport))
    sdk = openai.OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        max_retries=0,
        http_client=http,
    )
    return OpenAICompatibleClient(
        settings, client=sdk, sleep=(sleeps.append if sleeps is not None else lambda _: None)
    )


def ask(client: OpenAICompatibleClient, **kwargs: object):
    return client.complete(
        LLMRequest(
            system="static instruction",
            messages=(Message(role="user", content="the context"),),
            **kwargs,  # type: ignore[arg-type]
        )
    )


# --------------------------------------------------------------------------
# Request shape
# --------------------------------------------------------------------------


def test_request_carries_model_messages_and_limits() -> None:
    transport = Transport([completion("hello")])
    response = ask(build(make_settings(), transport))

    body = transport.body()
    assert body["model"] == "test-model"
    assert body["messages"] == [
        {"role": "system", "content": "static instruction"},
        {"role": "user", "content": "the context"},
    ]
    assert body["max_tokens"] == 2000
    assert "response_format" not in body
    assert response.text == "hello"
    assert (response.tokens_in, response.tokens_out) == (12, 7)
    assert response.attempts == 1


def test_json_mode_asks_for_a_json_object() -> None:
    transport = Transport([completion("{}")])
    ask(build(make_settings(), transport), json_mode=True)
    assert transport.body()["response_format"] == {"type": "json_object"}


def test_json_mode_can_be_refused_by_configuration() -> None:
    transport = Transport([completion("{}")])
    ask(build(make_settings(LLM_JSON_MODE="false"), transport), json_mode=True)
    assert "response_format" not in transport.body()


def test_max_tokens_comes_from_settings_and_can_be_overridden() -> None:
    transport = Transport([completion("x"), completion("y")])
    client = build(make_settings(LLM_MAX_TOKENS="321"), transport)
    ask(client)
    ask(client, max_tokens=99)
    assert transport.body(0)["max_tokens"] == 321
    assert transport.body(1)["max_tokens"] == 99


def test_tools_are_sent_as_function_specs() -> None:
    transport = Transport([completion("")])
    tool = ToolSpec(
        name="execute_sql",
        description="Run one read-only query.",
        parameters={"type": "object", "properties": {"sql": {"type": "string"}}},
    )
    ask(build(make_settings(), transport), tools=(tool,))

    wire = transport.body()["tools"]
    assert wire == [
        {
            "type": "function",
            "function": {
                "name": "execute_sql",
                "description": "Run one read-only query.",
                "parameters": {
                    "type": "object",
                    "properties": {"sql": {"type": "string"}},
                },
            },
        }
    ]


def test_a_tool_result_turn_round_trips() -> None:
    transport = Transport([completion("done")])
    call = ToolCall(id="call_1", name="execute_sql", arguments={"sql": "SELECT 1"})
    request = LLMRequest(
        system="sys",
        messages=(
            Message(role="user", content="go"),
            Message(role="assistant", tool_calls=(call,)),
            Message(role="tool", content="1", tool_call_id="call_1", name="execute_sql"),
        ),
    )
    build(make_settings(), transport).complete(request)

    messages = transport.body()["messages"]
    assert messages[2]["content"] is None
    assert messages[2]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "execute_sql", "arguments": '{"sql": "SELECT 1"}'},
        }
    ]
    assert messages[3] == {
        "role": "tool",
        "content": "1",
        "tool_call_id": "call_1",
        "name": "execute_sql",
    }


def test_base_url_is_honoured() -> None:
    transport = Transport([completion("x")])
    ask(build(make_settings(LLM_BASE_URL="https://other.test/openai/v1"), transport))
    request = transport.requests[0]
    assert request.url.host == "other.test"
    assert request.url.path == "/openai/v1/chat/completions"


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------


def test_tool_calls_are_parsed() -> None:
    transport = Transport(
        [
            completion(
                "",
                tool_calls=[
                    {
                        "id": "call_9",
                        "type": "function",
                        "function": {
                            "name": "execute_sql",
                            "arguments": '{"sql": "SELECT 1"}',
                        },
                    }
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    response = ask(build(make_settings(), transport))

    assert response.text == ""
    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert (call.id, call.name) == ("call_9", "execute_sql")
    assert call.arguments == {"sql": "SELECT 1"}
    assert call.parse_error == ""


def test_malformed_tool_arguments_are_reported_not_raised() -> None:
    transport = Transport(
        [
            completion(
                "",
                tool_calls=[
                    {
                        "id": "call_9",
                        "type": "function",
                        "function": {"name": "execute_sql", "arguments": "{oops"},
                    }
                ],
            )
        ]
    )
    call = ask(build(make_settings(), transport)).tool_calls[0]
    assert call.arguments == {}
    assert call.raw_arguments == "{oops"
    assert "not valid JSON" in call.parse_error


def test_non_object_tool_arguments_are_reported() -> None:
    transport = Transport(
        [
            completion(
                "",
                tool_calls=[
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "f", "arguments": "[1, 2]"},
                    }
                ],
            )
        ]
    )
    call = ask(build(make_settings(), transport)).tool_calls[0]
    assert call.arguments == {}
    assert "not a JSON object" in call.parse_error


def test_missing_usage_is_zero_not_crash() -> None:
    transport = Transport([completion("x", usage=None)])
    response = ask(build(make_settings(), transport))
    assert (response.tokens_in, response.tokens_out) == (0, 0)


def test_empty_choices_is_a_typed_error() -> None:
    transport = Transport([httpx.Response(200, json={"choices": []})])
    with pytest.raises(LLMError) as caught:
        ask(build(make_settings(), transport))
    assert caught.value.kind == "empty_response"
    assert caught.value.retryable is False


# --------------------------------------------------------------------------
# Failures
# --------------------------------------------------------------------------


def test_auth_failure_is_not_retried() -> None:
    transport = Transport(
        [httpx.Response(401, json={"error": {"message": "invalid api key"}})]
    )
    with pytest.raises(LLMError) as caught:
        ask(build(make_settings(), transport))

    assert caught.value.kind == "auth"
    assert caught.value.retryable is False
    assert caught.value.attempts == 1
    assert transport.count == 1


def test_rate_limit_is_retried_then_succeeds() -> None:
    sleeps: list[float] = []
    transport = Transport(
        [
            httpx.Response(429, json={"error": {"message": "slow down"}}),
            completion("finally"),
        ]
    )
    response = ask(build(make_settings(), transport, sleeps=sleeps))

    assert response.text == "finally"
    assert response.attempts == 2
    assert transport.count == 2
    assert sleeps == [0.5]


def test_retries_are_bounded_and_the_last_error_wins() -> None:
    transport = Transport(
        [
            httpx.Response(500, json={"error": {"message": "boom"}}),
            httpx.Response(500, json={"error": {"message": "boom again"}}),
            httpx.Response(500, json={"error": {"message": "boom third"}}),
            httpx.Response(500, json={"error": {"message": "boom fourth"}}),
        ]
    )
    with pytest.raises(LLMError) as caught:
        ask(build(make_settings(), transport))

    assert caught.value.kind == "server"
    assert caught.value.retryable is True
    assert caught.value.attempts == 4
    assert transport.count == 4


def test_retries_can_be_disabled() -> None:
    transport = Transport(
        [httpx.Response(429, json={"error": {"message": "slow down"}})]
    )
    with pytest.raises(LLMError) as caught:
        ask(build(make_settings(LLM_MAX_RETRIES="0"), transport))
    assert caught.value.attempts == 1
    assert transport.count == 1


def test_timeout_is_retryable() -> None:
    transport = Transport([httpx.ReadTimeout("too slow")])
    with pytest.raises(LLMError) as caught:
        ask(build(make_settings(LLM_MAX_RETRIES="0"), transport))
    assert caught.value.kind == "timeout"
    assert caught.value.retryable is True


def test_connection_failure_is_retryable() -> None:
    transport = Transport([httpx.ConnectError("refused")])
    with pytest.raises(LLMError) as caught:
        ask(build(make_settings(LLM_MAX_RETRIES="0"), transport))
    assert caught.value.kind == "connection"
    assert caught.value.retryable is True


def test_unknown_model_is_not_retried() -> None:
    transport = Transport(
        [httpx.Response(404, json={"error": {"message": "model not found"}})]
    )
    with pytest.raises(LLMError) as caught:
        ask(build(make_settings(), transport))
    assert caught.value.kind == "not_found"
    assert "model not found" in caught.value.message


def test_bad_request_is_not_retried() -> None:
    transport = Transport(
        [httpx.Response(400, json={"error": {"message": "max_tokens is unsupported"}})]
    )
    with pytest.raises(LLMError) as caught:
        ask(build(make_settings(), transport))
    assert caught.value.kind == "bad_request"
    assert caught.value.retryable is False
    assert transport.count == 1


def test_unexpected_client_error_status_is_not_retried() -> None:
    transport = Transport([httpx.Response(418, json={"error": {"message": "teapot"}})])
    with pytest.raises(LLMError) as caught:
        ask(build(make_settings(), transport))
    assert caught.value.kind == "api_error"
    assert caught.value.status == 418


def test_the_api_key_never_reaches_an_error_message() -> None:
    transport = Transport(
        [
            httpx.Response(
                400,
                json={"error": {"message": f"rejected credentials {API_KEY}"}},
            )
        ]
    )
    with pytest.raises(LLMError) as caught:
        ask(build(make_settings(), transport))

    assert API_KEY not in caught.value.message
    assert "***" in caught.value.message
