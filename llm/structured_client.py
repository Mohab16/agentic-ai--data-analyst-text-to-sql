"""A model-client adapter for models that cannot call tools natively.

The investigator's loop asks for `tool_calls` and feeds the results back. A model
without tool calling cannot do that, so this adapter rewrites the request: the
tool specs become a JSON schema in the system prompt, the model is asked for one
JSON object, and the adapter parses a `tool_calls` array out of it. The rest of
the loop -- the message shape, the tool results, the budgets, the findings turn
and its one repair -- is unchanged, which is the point (D-17): phase 6 adds a
caller rather than a second serialization.

The adapter is transparent for requests that carry no tools, so the planner,
the reporter, and the investigator's repair turn all behave exactly as before.
Only the investigator ever sends tools, so only it is affected.

The reply format is one JSON object:

    {"tool_calls": [{"id": "call_1", "name": "execute_sql",
                     "arguments": {"sql": "SELECT ..."}}]}

A reply with no `tool_calls` key is the model's findings, and the adapter returns
an empty tuple so the investigator's stop condition works unchanged. A reply
that is not valid JSON also returns an empty tuple: the investigator will treat
it as a findings attempt and its single repair turn will ask for the JSON it
actually needs.
"""

from __future__ import annotations

import json
from typing import Any

from llm.base import LLMClient, LLMRequest, LLMResponse, ToolCall, ToolSpec

_CALL_ID = "call_{}"
_TOOL_ASK = """\
When you want to use a tool, reply with exactly one JSON object and nothing else:

{{"tool_calls": [{{"id": "call_1", "name": "{names}", "arguments": {{...}}}}]}}

- `id` is any string you choose; it must be unique within this reply.
- `name` must be one of the tools listed below.
- `arguments` is the JSON object that tool's schema describes.

When you have enough evidence and want to give your findings, reply with a JSON
object that has no `tool_calls` key (or an empty list), as described in your
instructions.

The tools you may call, with the JSON Schema each one expects:

{schemas}
"""


def _schema_block(tool: ToolSpec) -> str:
    body = json.dumps(tool.parameters, ensure_ascii=False)
    return f"- {tool.name}: {tool.description}\n  arguments: {body}"


class StructuredToolClient:
    """Satisfies `LLMClient` by translating tool requests into JSON requests."""

    def __init__(self, delegate: LLMClient) -> None:
        self._delegate = delegate

    def complete(self, request: LLMRequest) -> LLMResponse:
        if not request.tools:
            return self._delegate.complete(request)
        return self._complete_with_tools(request)

    def _complete_with_tools(self, request: LLMRequest) -> LLMResponse:
        schemas = "\n".join(_schema_block(tool) for tool in request.tools)
        system = request.system + "\n\n" + _TOOL_ASK.format(
            names=", ".join(tool.name for tool in request.tools),
            schemas=schemas,
        )
        response = self._delegate.complete(
            LLMRequest(
                system=system,
                messages=request.messages,
                tools=(),
                json_mode=True,
                max_tokens=request.max_tokens,
            )
        )
        tool_calls = _parse_tool_calls(response.text, request.tools)
        return LLMResponse(
            text=response.text,
            model=response.model,
            tool_calls=tool_calls,
            finish_reason=response.finish_reason,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            attempts=response.attempts,
        )


def _parse_tool_calls(
    text: str, tools: tuple[ToolSpec, ...]
) -> tuple[ToolCall, ...]:
    """Extract a `tool_calls` array from a JSON reply.

    A reply with no tool calls (the model's findings) returns an empty tuple, so
    the investigator's stop condition works unchanged. A reply that is not valid
    JSON also returns an empty tuple: the investigator will treat it as a
    findings attempt and its one repair turn will ask for the JSON it needs.
    """
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return ()
    if not isinstance(payload, dict):
        return ()
    raw_calls = payload.get("tool_calls")
    if not isinstance(raw_calls, list) or not raw_calls:
        return ()
    names = {tool.name for tool in tools}
    calls: list[ToolCall] = []
    for index, raw in enumerate(raw_calls):
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        if not isinstance(name, str) or name not in names:
            continue
        arguments = raw.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(
            ToolCall(
                id=str(raw.get("id") or _CALL_ID.format(index + 1)),
                name=name,
                arguments=arguments,
                raw_arguments=json.dumps(arguments, ensure_ascii=False),
            )
        )
    return tuple(calls)