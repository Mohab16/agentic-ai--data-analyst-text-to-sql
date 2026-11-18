"""The investigator runs end-to-end through the structured-JSON adapter.

With `llm_supports_tool_calling=false` the client is wrapped, so the model's
native tool calls are replaced by a JSON object containing a `tool_calls` array.
The investigator's loop, budgets, and findings turn are all unchanged; this test
confirms the adapter is transparent enough that the same report shape comes out.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.investigator import make_investigator
from core.config import Settings
from llm.base import LLMError, LLMRequest, LLMResponse, ToolCall
from llm.structured_client import StructuredToolClient
from tools.registry import build_toolset

COUNT_SQL = "SELECT COUNT(*) AS completed_orders FROM orders WHERE status = 'completed'"

FINDINGS_REPLY = json.dumps(
    {
        "findings": [
            {
                "id": "f1",
                "claim": "Completed orders are counted in the query.",
                "evidence_query_ids": ["q_001"],
                "numbers": {"orders": 2797},
            }
        ],
        "open_questions": [],
        "notes": [],
    }
)


def _settings(env: dict[str, str], tmp_path: Path) -> Settings:
    return Settings.from_env(
        {
            **env,
            "TRACE_DIR": str(tmp_path / "traces"),
            "LLM_SUPPORTS_TOOL_CALLING": "false",
        }
    )


def _client(replies: list[object]) -> StructuredToolClient:
    """A scripted delegate wrapped exactly as `wire.build_agents` does it."""

    class _Delegate:
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

    return StructuredToolClient(_Delegate())


def test_investigator_runs_through_json_tool_calls(
    env: dict[str, str], tmp_path: Path
) -> None:
    settings = _settings(env, tmp_path)
    database = _build_database(settings)
    toolset = build_toolset(database, settings)

    # First turn: the model emits a JSON object with one tool call.
    tool_call_reply = json.dumps(
        {
            "tool_calls": [
                {"id": "call_1", "name": "execute_sql", "arguments": {"sql": COUNT_SQL}}
            ]
        }
    )
    client = _client([tool_call_reply, FINDINGS_REPLY])
    investigator = make_investigator(client, toolset)

    result = investigator("count completed orders", settings)

    assert result.status == "ok"
    assert result.data is not None
    report = result.data
    assert report["stop_reason"] == "model_finished"
    assert len(report["queries"]) == 1
    assert report["queries"][0]["status"] == "ok"
    assert report["queries"][0]["id"] == "q_001"
    assert len(report["findings"]) == 1
    assert report["findings"][0]["evidence_query_ids"] == ["q_001"]


def test_investigator_stop_on_findings_without_tool_calls(
    env: dict[str, str], tmp_path: Path
) -> None:
    settings = _settings(env, tmp_path)
    database = _build_database(settings)
    toolset = build_toolset(database, settings)

    # The model never calls a tool: it reports what it cannot answer. A claim
    # without evidence is rejected by the parser, so this reply cites none.
    no_evidence_reply = json.dumps(
        {
            "findings": [],
            "open_questions": ["the schema digest does not say how many orders exist"],
            "notes": ["run execute_sql to count them"],
        }
    )
    client = _client([no_evidence_reply])
    investigator = make_investigator(client, toolset)

    result = investigator("count completed orders", settings)

    assert result.status == "ok"
    assert result.data["stop_reason"] == "model_finished"
    assert len(result.data["queries"]) == 0
    assert result.data["open_questions"]


def _build_database(settings: Settings):
    from db.factory import build_database

    return build_database(settings)