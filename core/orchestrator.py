"""The skeleton: plan construction, validation, dispatch, and the step loop.

This module is the only owner of plan-to-execution (L-03, O-01, O-02). It knows
nothing about SQL, prompts, HTTP, or rendering.

Rules: C-02 (agent keys, never prose), C-04 (validate before the first step),
O-03 (explicit budgets), S-01/S-02 (one writer of task state), R-01 (one trace
event per step).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256

from core.config import Settings
from core.context import build_context
from core.trace import TraceSink
from core.types import (
    AgentResult,
    PlanInvalid,
    Plan,
    Step,
    TaskState,
    TraceEvent,
)

# The three roles of this workflow, by cognitive function (A-02, A-03).
# Phase 3 implements the planner; phases 4 and 5 add the rest. Wire exposes only
# what exists, and a test asserts it never claims more.
AGENT_KEYS: tuple[str, ...] = ("planner", "investigator", "reporter")

# Artifacts present before the first step: the question and the schema digest.
ROOT_ARTIFACTS: tuple[str, ...] = ("question", "schema")

AgentFn = Callable[[str, Settings], AgentResult]


@dataclass(frozen=True)
class RunOutcome:
    """Everything a transport needs after a run. State is derived, not mirrored."""

    state: TaskState
    artifacts: dict[str, AgentResult]


def build_plan(task_id: str, *, max_steps: int = 12) -> Plan:
    """The workflow shape. Fixed and inspectable; autonomy lives inside a step."""
    steps = (
        Step(
            id="plan",
            agent="planner",
            instruction=(
                "Turn the business question into an investigation plan: define the "
                "metric, resolve the time period to concrete dates, state the "
                "comparison baseline, list the checks worth running, and record "
                "assumptions. If the schema cannot answer the question, say so."
            ),
            reads=("question", "schema"),
        ),
        Step(
            id="investigate",
            agent="investigator",
            instruction=(
                "Produce evidence for the question by writing and executing SQL. "
                "Use the schema artifact for structure and inspect_schema for detail. "
                "Every finding must cite the query ids that support it. Stop when the "
                "evidence is sufficient; report gaps rather than guessing."
            ),
            reads=("question", "plan", "schema"),
        ),
        Step(
            id="answer",
            agent="reporter",
            instruction=(
                "Write the business answer from the investigation report. Use only "
                "numbers present in the evidence, keep correlation and causation "
                "distinct, and state assumptions and limitations. Do not emit SQL: "
                "the caller renders the executed queries itself."
            ),
            reads=("question", "plan", "investigate"),
        ),
    )
    return Plan(task_id=task_id, steps=steps, max_steps=max_steps)


def validate_plan(
    plan: Plan,
    known_agents: Collection[str],
    roots: Collection[str] = ROOT_ARTIFACTS,
) -> None:
    """Reject a malformed plan before the first step runs (C-04)."""
    problems: list[str] = []

    if not plan.steps:
        raise PlanInvalid(["plan has no steps"])
    if len(plan.steps) > plan.max_steps:
        problems.append(
            f"plan has {len(plan.steps)} steps, max_steps is {plan.max_steps}"
        )

    seen: set[str] = set(roots)
    for index, step in enumerate(plan.steps):
        where = f"step[{index}] ({step.id or 'no id'})"
        if not step.id:
            problems.append(f"{where}: id is empty")
        elif step.id in seen:
            problems.append(f"{where}: id is duplicated")
        elif not _is_slug(step.id):
            problems.append(f"{where}: id must match [a-z][a-z0-9_]*")
        if not step.agent:
            problems.append(f"{where}: agent is empty")
        elif step.agent not in known_agents:
            problems.append(f"{where}: unknown agent {step.agent!r}")
        if not step.instruction.strip():
            problems.append(f"{where}: instruction is empty")
        for read in step.reads:
            if read not in seen:
                problems.append(f"{where}: reads undefined artifact {read!r}")
        if len(set(step.reads)) != len(step.reads):
            problems.append(f"{where}: reads contains duplicates")
        seen.add(step.id)

    if problems:
        raise PlanInvalid(problems)


def select_steps(plan: Plan, step_ids: Collection[str]) -> Plan:
    """Keep only the named steps, in workflow order.

    A prefix run (`--plan`) uses the real workflow rather than a parallel one, so
    the step, its declared reads, and its trace event are identical to a full run.
    """
    kept = tuple(step for step in plan.steps if step.id in step_ids)
    if not kept:
        raise PlanInvalid([f"no step of this plan is in {sorted(step_ids)}"])
    return replace(plan, steps=kept)


def execute(
    plan: Plan,
    agents: Mapping[str, AgentFn],
    settings: Settings,
    trace: TraceSink,
    roots: Mapping[str, AgentResult],
) -> RunOutcome:
    """Run the steps in order. Stop on the first agent error (O-02, E-01)."""
    validate_plan(plan, set(agents), set(roots))

    artifacts: dict[str, AgentResult] = dict(roots)
    state = TaskState(task_id=plan.task_id, status="running")
    started = time.monotonic()

    for step in plan.steps:
        elapsed = time.monotonic() - started
        if elapsed > settings.wall_clock_s:
            return RunOutcome(
                state=replace(
                    state,
                    status="error",
                    current_step=step.id,
                    error=(
                        f"budget_exceeded: wall_clock_s={settings.wall_clock_s} "
                        f"after {elapsed:.1f}s"
                    ),
                ),
                artifacts=artifacts,
            )

        context, truncated = build_context(step, artifacts, settings.context_budget)
        state = replace(state, current_step=step.id)

        step_started = time.monotonic()
        started_at = datetime.now(UTC).isoformat(timespec="seconds")
        result = _call_agent(agents[step.agent], context, settings)
        duration_ms = int((time.monotonic() - step_started) * 1000)

        trace.emit(
            TraceEvent(
                task_id=plan.task_id,
                step_id=step.id,
                agent=step.agent,
                model=settings.llm_model,
                started_at=started_at,
                duration_ms=duration_ms,
                outcome=_outcome(result, truncated),
                prompt_hash=_prompt_hash(step, context),
                tool_calls=result.tool_calls,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                error=result.error,
            )
        )

        artifacts[step.id] = result
        state = replace(state, steps_completed=state.steps_completed + 1)

        if result.status == "error":
            return RunOutcome(
                state=replace(
                    state,
                    status="error",
                    error=result.error or f"agent_error: {step.agent}",
                ),
                artifacts=artifacts,
            )

    return RunOutcome(
        state=replace(state, status="done", current_step=None),
        artifacts=artifacts,
    )


def _call_agent(agent: AgentFn, context: str, settings: Settings) -> AgentResult:
    """An agent that raises violates its contract; convert it to a typed error."""
    try:
        return agent(context, settings)
    except Exception as exc:  # noqa: BLE001 - contract boundary, recorded in trace
        return AgentResult(
            status="error",
            error=f"agent_raised: {type(exc).__name__}: {exc}",
        )


def _outcome(result: AgentResult, truncated: bool) -> str:
    if result.status == "error":
        return "error"
    if truncated:
        return "truncated"
    return "ok"


def _prompt_hash(step: Step, context: str) -> str:
    """A stable fingerprint of exactly what the agent saw, without storing it.

    Two runs that sent the same instruction and context share a hash, so a trace
    can be compared across runs without copying the prompt into the log.
    """
    material = f"{step.instruction}\n\n{context}".encode("utf-8")
    return sha256(material).hexdigest()[:16]


def _is_slug(value: str) -> bool:
    if not value or not value[0].isalpha() or not value[0].islower():
        return False
    return all(ch.islower() or ch.isdigit() or ch == "_" for ch in value)
