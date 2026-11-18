"""Composition root.

The one module that knows concrete implementations: it builds the database, the
model client, and the agent dispatch map, so `app/main.py` and `core/` never
import `db/`, `tools/`, `agents/`, or `llm/` directly (L-01). It contains no
logic of its own: every function here is a decision about which concrete object
to construct, in one place.
"""

from __future__ import annotations

from agents.investigator import make_investigator
from agents.planner import make_planner
from agents.reporter import make_reporter
from core.config import Settings
from core.orchestrator import AgentFn, RunOutcome, build_plan, execute, select_steps
from core.trace import TraceStore
from core.types import AgentResult, ToolError, new_task_id
from db.base import Database, QueryResult
from db.factory import build_database
from llm.base import LLMClient
from llm.openai_client import OpenAICompatibleClient
from llm.structured_client import StructuredToolClient
from tools.registry import build_toolset
from tools.schema import make_inspect_schema_tool, render_schema_digest
from tools.sql import (
    make_error_details_tool,
    make_execute_sql_tool,
    render_query_result,
)


def open_database(settings: Settings) -> Database:
    """Build the configured backend. Raises `BackendUnavailable` if it cannot serve."""
    return build_database(settings)


def schema_artifact(database: Database, settings: Settings) -> AgentResult:
    """The `schema` root artifact: a bounded digest of every table."""
    text, _truncated = render_schema_digest(database.schema(), settings.context_budget)
    return AgentResult(status="ok", text=text)


def inspect_table(database: Database, table: str) -> str | ToolError:
    """One table in full detail, through the same code path the model will use."""
    return make_inspect_schema_tool(database)(table)


def execute_sql(
    database: Database, settings: Settings, sql: str
) -> QueryResult | ToolError:
    """Run one query through the guard and the executor, as the model would."""
    return make_execute_sql_tool(database, settings)(sql)


def query_error_details(database: Database, error_id: str) -> str | ToolError:
    """Full details for a failed query, as the model would ask for them."""
    return make_error_details_tool(database)(error_id)


def render_result(result: QueryResult) -> str:
    """Text form of a result, for the CLI."""
    return render_query_result(result)


def build_llm(settings: Settings) -> LLMClient:
    """The model client for one run: it owns attempts, so they can be traced."""
    return OpenAICompatibleClient(settings)


def build_agents(settings: Settings, database: Database) -> dict[str, AgentFn]:
    """Every agent the workflow declares: planner, investigator, reporter.

    One client and one toolset per run: the toolset holds the query budget and the
    evidence records, so it must not outlive or be shared across runs.

    A model that cannot call tools natively still has to run the investigator's
    loop, so the client is wrapped in `StructuredToolClient` when
    `llm_supports_tool_calling` is false. The planner and reporter never send
    tools, so the adapter is transparent for them.
    """
    client: LLMClient = build_llm(settings)
    if not settings.llm_supports_tool_calling:
        client = StructuredToolClient(client)
    toolset = build_toolset(database, settings)
    return {
        "planner": make_planner(client),
        "investigator": make_investigator(client, toolset),
        "reporter": make_reporter(client),
    }


def _run(
    settings: Settings, question: str, step_ids: tuple[str, ...]
) -> RunOutcome:
    """Run the named steps of the fixed workflow, on one task and one trace."""
    database = open_database(settings)
    artifacts = {
        "question": AgentResult(status="ok", text=question),
        "schema": schema_artifact(database, settings),
    }
    plan = select_steps(
        build_plan(new_task_id(), max_steps=settings.max_steps), step_ids
    )
    return execute(
        plan,
        build_agents(settings, database),
        settings,
        TraceStore.from_settings(settings),
        artifacts,
    )


def plan_question(settings: Settings, question: str) -> RunOutcome:
    """Run the workflow's first step for real: question + schema in, plan out.

    Going through `execute` rather than calling the agent directly keeps the
    trace, the context budget, and the wall-clock check on the same path a full
    run uses.
    """
    return _run(settings, question, ("plan",))


def investigate_question(settings: Settings, question: str) -> RunOutcome:
    """Plan, then investigate. The evidence, without the answer around it."""
    return _run(settings, question, ("plan", "investigate"))


def answer_question(settings: Settings, question: str) -> RunOutcome:
    """The whole workflow: plan, investigate, report."""
    return _run(settings, question, ("plan", "investigate", "answer"))
