"""The structured-JSON tool-calling adapter (phase 6).

A model that cannot call tools natively still has to run the investigator's
bounded loop. `StructuredToolClient` rewrites a tool request into a JSON-schema
request and parses a `tool_calls` array out of the reply, so the investigator's
message shape, budgets, and findings turn are all unchanged.

These tests are the offline stand-in for the real provider: the delegate is a
plain function, and the assertions are about the translation, not about a model.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm.base import LLMClient, LLMError, LLMRequest, LLMResponse, ToolCall, ToolSpec
from llm.structured_client import StructuredToolClient

_CALL = ToolSpec(
    name="execute_sql",
    description="run one guarded SELECT",
    parameters={
        "type": "object",
        "properties": {"sql": {"type": "string"}},
        "required": ["sql"],
    },
)


def _delegate(replies: list[object]) -> tuple[LLMClient, list[LLMRequest]]:
    """A stand-in client that records requests and returns replies in order."""

    class _Client:
        def __init__(self) -> None:
            self.requests: list[LLMRequest] = []

        def complete(self, request: LLMRequest) -> LLMResponse:
            self.requests.append(request)
            if not replies:
                raise AssertionError("delegate has no replies left")
            reply = replies.pop(0)
            if isinstance(reply, LLMError):
                raise reply
            if isinstance(reply, str):
                return LLMResponse(text=reply, model="scripted")
            return reply  # type: ignore[return-value]

    client = _Client()
    return client, client.requests


def _request(tools: tuple[ToolSpec, ...] = ()) -> LLMRequest:
    return LLMRequest(
        system="you are the investigator",
        messages=(),
        tools=tools,
    )


# --------------------------------------------------------------------------


def test_passthrough_when_no_tools() -> None:
    delegate, requests = _delegate([LLMResponse(text="plain", model="m")])
    client = StructuredToolClient(delegate)

    response = client.complete(_request())

    assert response.text == "plain"
    assert response.tool_calls == ()
    assert len(requests) == 1
    assert requests[0].tools == ()
    assert requests[0].json_mode is False


def test_json_with_tool_calls_is_parsed() -> None:
    reply = json.dumps(
        {
            "tool_calls": [
                {
                    "id": "c1",
                    "name": "execute_sql",
                    "arguments": {"sql": "SELECT 1"},
                }
            ]
        }
    )
    delegate, _ = _delegate([reply])
    client = StructuredToolClient(delegate)

    response = client.complete(_request(tools=(_CALL,)))

    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.id == "c1"
    assert call.name == "execute_sql"
    assert call.arguments == {"sql": "SELECT 1"}
    assert call.parse_error == ""


def test_reply_with_no_tool_calls_is_the_findings_turn() -> None:
    reply = json.dumps({"findings": [], "open_questions": [], "notes": []})
    delegate, _ = _delegate([reply])
    client = StructuredToolClient(delegate)

    response = client.complete(_request(tools=(_CALL,)))

    assert response.tool_calls == ()


def test_invalid_json_returns_no_tool_calls() -> None:
    delegate, _ = _delegate(["not json at all"])
    client = StructuredToolClient(delegate)

    response = client.complete(_request(tools=(_CALL,)))

    assert response.tool_calls == ()


def test_unknown_tool_name_is_dropped() -> None:
    reply = json.dumps(
        {
            "tool_calls": [
                {
                    "id": "c1",
                    "name": "inspect_schema",
                    "arguments": {"table": "orders"},
                },
                {
                    "id": "c2",
                    "name": "execute_sql",
                    "arguments": {"sql": "SELECT 1"},
                },
            ]
        }
    )
    delegate, _ = _delegate([reply])
    client = StructuredToolClient(delegate)

    response = client.complete(_request(tools=(_CALL,)))

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "c2"


def test_non_dict_arguments_become_empty() -> None:
    reply = json.dumps(
        {"tool_calls": [{"id": "c1", "name": "execute_sql", "arguments": "oops"}]}
    )
    delegate, _ = _delegate([reply])
    client = StructuredToolClient(delegate)

    response = client.complete(_request(tools=(_CALL,)))

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].arguments == {}


def test_request_carries_json_mode_and_schema() -> None:
    reply = json.dumps({"tool_calls": []})
    delegate, requests = _delegate([reply])
    client = StructuredToolClient(delegate)

    client.complete(_request(tools=(_CALL,)))

    request = requests[0]
    assert request.tools == ()
    assert request.json_mode is True
    assert "execute_sql" in request.system
    assert "SELECT 1" not in request.system  # the schema, not a value
    assert '"sql"' in request.system  # the JSON schema for the argument


def test_delegate_error_propagates() -> None:
    delegate, _ = _delegate([LLMError("rate_limit", "slow down", retryable=True)])
    client = StructuredToolClient(delegate)

    with pytest.raises(LLMError) as caught:
        client.complete(_request(tools=(_CALL,)))

    assert caught.value.kind == "rate_limit"