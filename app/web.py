"""FastAPI web server: HTTP transport for the agentic workflow.

Rules: L-01 (transport layer knows nothing about agents), P-01 (rendering at
the edge), 13.8 (LLM calls run in a thread, never blocking the event loop).

The API is deliberately thin: it hands the question to `wire.*`, which owns the
composition root, and returns JSON. No HTML, no UI affordances, no prompts.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load .env automatically for the FastAPI process (same as the CLI)
from dotenv import load_dotenv

load_dotenv()

import wire
from core.config import Settings
from core.types import ConfigError, PlanInvalid, to_dict

app = FastAPI(
    title="AI Data Analyst",
    description="Agentic text-to-SQL analyst: the LLM investigates a database through validated SQL.",
    version="0.1.0",
)


class QuestionRequest(BaseModel):
    """A business question to investigate."""

    question: str = Field(..., min_length=1, max_length=2000)


class PlanRequest(BaseModel):
    """A business question; returns the investigation plan only."""

    question: str = Field(..., min_length=1, max_length=2000)


class InvestigateRequest(BaseModel):
    """A business question; returns plan + evidence, no answer."""

    question: str = Field(..., min_length=1, max_length=2000)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str


class AnswerResponse(BaseModel):
    """Full answer response: plan, investigation, answer, and run facts."""

    task_id: str
    question: str
    plan: dict[str, Any] | None = None
    investigation: dict[str, Any] | None = None
    answer: dict[str, Any] | None = None
    tokens_in: int
    tokens_out: int
    settings: dict[str, Any]


class PlanResponse(BaseModel):
    """Plan-only response."""

    task_id: str
    question: str
    plan: dict[str, Any]
    summary: str
    tokens_in: int
    tokens_out: int
    settings: dict[str, Any]


class InvestigateResponse(BaseModel):
    """Investigation-only response: plan + evidence."""

    task_id: str
    question: str
    plan: dict[str, Any] | None
    investigation: dict[str, Any] | None
    tokens_in: int
    tokens_out: int
    settings: dict[str, Any]


def _get_settings() -> Settings:
    """Build settings for this request. Raises HTTPException on config error."""
    try:
        return Settings.from_env(require_llm=True)
    except ConfigError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "configuration_error",
                "problems": list(exc.problems),
            },
        ) from None


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check: returns status and version."""
    return HealthResponse(status="ok", version="0.1.0")


@app.post("/answer", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest) -> AnswerResponse:
    """Full workflow: plan, investigate, and answer.

    The LLM generates an investigation plan, executes SQL through the guard,
    and writes a business answer. All evidence is cited.
    """
    settings = _get_settings()
    loop = asyncio.get_running_loop()

    try:
        # Run the blocking call in a thread (13.8): never block the event loop
        outcome = await asyncio.to_thread(wire.answer_question, settings, request.question)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "execution_failed", "message": str(exc)},
        ) from exc

    state = outcome.state
    if state.status != "done":
        raise HTTPException(
            status_code=500,
            detail={
                "error": "run_failed",
                "status": state.status,
                "error_message": state.error,
            },
        )

    plan = outcome.artifacts.get("plan")
    investigation = outcome.artifacts.get("investigate")
    answer = outcome.artifacts.get("answer")

    def _tokens(field: str) -> int:
        return sum(
            getattr(result, field, 0)
            for result in [plan, investigation, answer]
            if result is not None
        )

    return AnswerResponse(
        task_id=state.task_id,
        question=request.question,
        plan=to_dict(plan.data) if plan and plan.data else None,
        investigation=to_dict(investigation.data) if investigation and investigation.data else None,
        answer=to_dict(answer.data) if answer and answer.data else None,
        tokens_in=_tokens("tokens_in"),
        tokens_out=_tokens("tokens_out"),
        settings=settings.public_config(),
    )


@app.post("/plan", response_model=PlanResponse)
async def plan_question(request: PlanRequest) -> PlanResponse:
    """One model call: question + schema in, validated plan out.

    Returns the investigation plan without executing any SQL.
    """
    settings = _get_settings()
    loop = asyncio.get_running_loop()

    try:
        outcome = await asyncio.to_thread(wire.plan_question, settings, request.question)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "execution_failed", "message": str(exc)},
        ) from exc

    state = outcome.state
    if state.status != "done":
        raise HTTPException(
            status_code=500,
            detail={
                "error": "run_failed",
                "status": state.status,
                "error_message": state.error,
            },
        )

    result = outcome.artifacts["plan"]
    return PlanResponse(
        task_id=state.task_id,
        question=request.question,
        plan=to_dict(result.data) if result.data else {},
        summary=result.text,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        settings=settings.public_config(),
    )


@app.post("/investigate", response_model=InvestigateResponse)
async def investigate_question(request: InvestigateRequest) -> InvestigateResponse:
    """Plan, then investigate. Returns evidence without the answer.

    Executes SQL through the guard and returns the evidence trail.
    """
    settings = _get_settings()
    loop = asyncio.get_running_loop()

    try:
        outcome = await asyncio.to_thread(wire.investigate_question, settings, request.question)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "execution_failed", "message": str(exc)},
        ) from exc

    state = outcome.state
    if state.status != "done":
        raise HTTPException(
            status_code=500,
            detail={
                "error": "run_failed",
                "status": state.status,
                "error_message": state.error,
            },
        )

    plan = outcome.artifacts.get("plan")
    investigation = outcome.artifacts.get("investigate")

    def _tokens(field: str) -> int:
        return sum(
            getattr(result, field, 0)
            for result in [plan, investigation]
            if result is not None
        )

    return InvestigateResponse(
        task_id=state.task_id,
        question=request.question,
        plan=to_dict(plan.data) if plan and plan.data else None,
        investigation=to_dict(investigation.data) if investigation and investigation.data else None,
        tokens_in=_tokens("tokens_in"),
        tokens_out=_tokens("tokens_out"),
        settings=settings.public_config(),
    )
