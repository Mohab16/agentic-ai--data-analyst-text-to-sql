"""CLI entry point.

Responsibilities: read argv, load `.env` (once, here: G-02), construct
`Settings` (once: G-01), build and validate the plan, render or dispatch.

It knows nothing about prompts or models: it asks `wire` for agents and renders
whatever comes back. SQL it does not interpret either: it hands the statement to
the guard and the tool layer, then decides an exit code.

The `--serve` flag starts the FastAPI server (uvicorn required).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from typing import Any

from dotenv import load_dotenv

import wire
from app.render import (
    render_answer_output,
    render_plan,
    render_planner_output,
    render_query_log,
    render_run_output,
)
from core.config import Settings
from core.orchestrator import AGENT_KEYS, ROOT_ARTIFACTS, build_plan, validate_plan
from core.types import (
    AgentResult,
    BackendUnavailable,
    ConfigError,
    PlanInvalid,
    ToolError,
    new_task_id,
    to_dict,
)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_CONFIG = 3
EXIT_PLAN = 4
EXIT_NOT_IMPLEMENTED = 5
EXIT_SQL_REJECTED = 6
EXIT_SQL_FAILED = 7
EXIT_RUN_FAILED = 8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description="Ask a business question; an LLM investigates the database with SQL.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help='e.g. "Why did our revenue drop this month?"',
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate configuration and print the plan without calling a model",
    )
    mode.add_argument(
        "--plan",
        action="store_true",
        help="call the model once and print the investigation plan",
    )
    mode.add_argument(
        "--investigate",
        action="store_true",
        help="run the plan and the investigation, and print the evidence without an answer",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of text",
    )
    parser.add_argument(
        "--trace-dir",
        metavar="DIR",
        help="override TRACE_DIR for this run",
    )
    parser.add_argument(
        "--show-schema",
        action="store_true",
        help="print the database schema digest the planner will receive",
    )
    parser.add_argument(
        "--inspect-table",
        metavar="NAME",
        help="print one table in full detail (columns, value domains, sample rows)",
    )
    parser.add_argument(
        "--query",
        metavar="SQL",
        help="execute one read-only statement through the guard and print the rows",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="with --query, also print full details when the query fails",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="start the FastAPI server (requires uvicorn)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address for --serve (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="port for --serve (default: 8000)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.serve:
        return _serve(args.host, args.port)

    load_dotenv()

    if args.explain and not args.query:
        print("error: --explain only applies to --query", file=sys.stderr)
        return EXIT_USAGE

    overrides: dict[str, str] = {}
    if args.trace_dir:
        overrides["TRACE_DIR"] = args.trace_dir

    needs_llm = not (
        args.dry_run or args.show_schema or args.inspect_table or args.query
    )
    try:
        settings = Settings.from_env({**os.environ, **overrides}, require_llm=needs_llm)
    except ConfigError as exc:
        print("configuration error:", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        print("see .env.example for the full list of variables", file=sys.stderr)
        return EXIT_CONFIG

    if args.show_schema or args.inspect_table:
        return _inspect(settings, show_all=args.show_schema, table=args.inspect_table)

    if args.query:
        return _run_query(settings, args.query, explain=args.explain)

    question = (args.question or "").strip()
    if not question:
        print("error: provide a question (see --help)", file=sys.stderr)
        return EXIT_USAGE

    if args.plan:
        return _run_plan(settings, question, as_json=args.json)

    if args.investigate:
        return _run_investigation(settings, question, as_json=args.json)

    if args.dry_run:
        plan = build_plan(new_task_id(), max_steps=settings.max_steps)
        try:
            validate_plan(plan, AGENT_KEYS, ROOT_ARTIFACTS)
        except PlanInvalid as exc:
            print("plan error:", file=sys.stderr)
            for problem in exc.problems:
                print(f"  - {problem}", file=sys.stderr)
            return EXIT_PLAN

        if args.json:
            print(
                json.dumps(
                    {
                        "task_id": plan.task_id,
                        "question": question,
                        "plan": to_dict(plan),
                        "settings": settings.public_config(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(render_plan(plan, settings, question))
        return EXIT_OK

    return _run_answer(settings, question, as_json=args.json)


def _run_answer(settings: Settings, question: str, *, as_json: bool) -> int:
    """The whole workflow: plan, investigate, and the answer a reader asked for."""
    try:
        outcome = wire.answer_question(settings, question)
    except BackendUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED

    state = outcome.state
    plan = outcome.artifacts.get("plan")
    investigation = outcome.artifacts.get("investigate")
    answer = outcome.artifacts.get("answer")

    if state.status != "done":
        print(f"error: {state.error or state.status}", file=sys.stderr)
        queries = _queries_of(investigation)
        log = render_query_log(queries)
        if log:
            print(log, file=sys.stderr)
        if as_json:
            print(
                json.dumps(
                    {
                        "task_id": state.task_id,
                        "question": question,
                        "error": state.error,
                        "queries": queries,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return EXIT_RUN_FAILED

    if as_json:
        print(
            json.dumps(
                {
                    "task_id": state.task_id,
                    "question": question,
                    "plan": plan.data if plan is not None else None,
                    "investigation": (
                        investigation.data if investigation is not None else None
                    ),
                    "answer": answer.data if answer is not None else None,
                    "tokens_in": _tokens(plan, investigation, answer, field="tokens_in"),
                    "tokens_out": _tokens(
                        plan, investigation, answer, field="tokens_out"
                    ),
                    "settings": settings.public_config(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(
            render_answer_output(
                answer,
                plan,
                investigation,
                task_id=state.task_id,
                settings=settings,
            )
        )
    return EXIT_OK


def _run_plan(settings: Settings, question: str, *, as_json: bool) -> int:
    """One model call: question + schema in, validated plan out."""
    try:
        outcome = wire.plan_question(settings, question)
    except BackendUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED

    state = outcome.state
    if state.status != "done":
        print(f"error: {state.error or state.status}", file=sys.stderr)
        return EXIT_RUN_FAILED

    result = outcome.artifacts["plan"]
    if as_json:
        print(
            json.dumps(
                {
                    "task_id": state.task_id,
                    "question": question,
                    "plan": result.data,
                    "summary": result.text,
                    "tokens_in": result.tokens_in,
                    "tokens_out": result.tokens_out,
                    "settings": settings.public_config(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(
            render_planner_output(result, task_id=state.task_id, settings=settings)
        )
    return EXIT_OK


def _run_investigation(settings: Settings, question: str, *, as_json: bool) -> int:
    """Plan, then investigate. Prints the evidence trail the answer will rest on."""
    try:
        outcome = wire.investigate_question(settings, question)
    except BackendUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED

    state = outcome.state
    plan = outcome.artifacts.get("plan")
    investigation = outcome.artifacts.get("investigate")

    if state.status != "done":
        print(f"error: {state.error or state.status}", file=sys.stderr)
        queries = _queries_of(investigation)
        log = render_query_log(queries)
        if log:
            print(log, file=sys.stderr)
        if as_json:
            print(
                json.dumps(
                    {
                        "task_id": state.task_id,
                        "question": question,
                        "error": state.error,
                        "queries": queries,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return EXIT_RUN_FAILED

    if as_json:
        print(
            json.dumps(
                {
                    "task_id": state.task_id,
                    "question": question,
                    "plan": plan.data if plan is not None else None,
                    "investigation": investigation.data if investigation is not None else None,
                    "tokens_in": _tokens(plan, investigation, "tokens_in"),
                    "tokens_out": _tokens(plan, investigation, "tokens_out"),
                    "settings": settings.public_config(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(
            render_run_output(
                plan, investigation, task_id=state.task_id, settings=settings
            )
        )
    return EXIT_OK


def _queries_of(investigation: AgentResult | None) -> list[dict[str, Any]]:
    if investigation is None or not investigation.data:
        return []
    queries = investigation.data.get("queries")
    return list(queries) if isinstance(queries, list) else []


def _tokens(*results: AgentResult | None, field: str) -> int:
    return sum(getattr(result, field, 0) for result in results if result is not None)


def _inspect(settings: Settings, *, show_all: bool, table: str | None) -> int:
    """Schema inspection: the same adapter and renderer the agent will use."""
    try:
        database = wire.open_database(settings)
    except BackendUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED

    if show_all:
        print(wire.schema_artifact(database, settings).text)

    if table is not None:
        result = wire.inspect_table(database, table)
        if isinstance(result, ToolError):
            print(f"error: {result.message}", file=sys.stderr)
            if result.detail:
                print(f"  {result.detail}", file=sys.stderr)
            return EXIT_USAGE
        print(result)

    return EXIT_OK


def _run_query(settings: Settings, sql: str, *, explain: bool) -> int:
    """One statement through the guard and the executor, exactly as the agent would."""
    try:
        database = wire.open_database(settings)
    except BackendUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED

    outcome = wire.execute_sql(database, settings, sql)
    if isinstance(outcome, ToolError):
        print(f"error: {outcome.message}", file=sys.stderr)
        if outcome.detail:
            print(f"  {outcome.detail}", file=sys.stderr)
        if explain:
            recent = database.recent_error_ids()
            if recent:
                details = wire.query_error_details(database, recent[-1])
                print(file=sys.stderr)
                print(
                    details if isinstance(details, str) else details.message,
                    file=sys.stderr,
                )
        return EXIT_SQL_REJECTED if outcome.kind == "rejected" else EXIT_SQL_FAILED

    print(wire.render_result(outcome))
    return EXIT_OK


def _serve(host: str, port: int) -> int:
    """Start the FastAPI server.

    The server runs indefinitely until interrupted. It returns EXIT_OK when the
    process exits cleanly. If uvicorn is not installed, we emit a helpful error
    and return EXIT_NOT_IMPLEMENTED.
    """
    try:
        import uvicorn
    except Exception:  # pragma: no cover – defensive import guard
        print(
            "uvicorn is required for the '--serve' option. Install it with 'pip install uvicorn'",
            file=sys.stderr,
        )
        return EXIT_NOT_IMPLEMENTED

    # Import inside the function to avoid importing FastAPI when the CLI is used.
    from app.web import app as fastapi_app

    uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
