"""A malformed plan is rejected before the first step runs (C-04)."""

from __future__ import annotations

import pytest

from core.orchestrator import (
    AGENT_KEYS,
    ROOT_ARTIFACTS,
    build_plan,
    select_steps,
    validate_plan,
)
from core.types import Plan, PlanInvalid, Step


def test_built_plan_is_valid() -> None:
    validate_plan(build_plan("tsk_1"), AGENT_KEYS, ROOT_ARTIFACTS)


def test_built_plan_uses_exactly_the_known_agents() -> None:
    plan = build_plan("tsk_1")
    assert [step.agent for step in plan.steps] == list(AGENT_KEYS)


def test_built_plan_steps_have_unique_ids() -> None:
    plan = build_plan("tsk_1")
    ids = [step.id for step in plan.steps]
    assert len(ids) == len(set(ids))


def test_built_plan_reads_only_known_artifacts() -> None:
    plan = build_plan("tsk_1")
    available = set(ROOT_ARTIFACTS)
    for step in plan.steps:
        assert set(step.reads) <= available
        available.add(step.id)


def test_empty_plan_is_rejected() -> None:
    with pytest.raises(PlanInvalid) as caught:
        validate_plan(Plan(task_id="t", steps=()), AGENT_KEYS, ROOT_ARTIFACTS)
    assert "no steps" in str(caught.value)


def test_unknown_agent_is_rejected() -> None:
    plan = Plan(
        task_id="t",
        steps=(Step(id="a", agent="summarizer", instruction="x"),),
    )
    with pytest.raises(PlanInvalid) as caught:
        validate_plan(plan, AGENT_KEYS, ROOT_ARTIFACTS)
    assert "unknown agent" in str(caught.value)


def test_duplicate_step_ids_are_rejected() -> None:
    plan = Plan(
        task_id="t",
        steps=(
            Step(id="a", agent="planner", instruction="x"),
            Step(id="a", agent="reporter", instruction="y"),
        ),
    )
    with pytest.raises(PlanInvalid) as caught:
        validate_plan(plan, AGENT_KEYS, ROOT_ARTIFACTS)
    assert "duplicated" in str(caught.value)


def test_dangling_read_is_rejected() -> None:
    plan = Plan(
        task_id="t",
        steps=(Step(id="a", agent="planner", instruction="x", reads=("ghost",)),),
    )
    with pytest.raises(PlanInvalid) as caught:
        validate_plan(plan, AGENT_KEYS, ROOT_ARTIFACTS)
    assert "ghost" in str(caught.value)


def test_forward_read_is_rejected() -> None:
    plan = Plan(
        task_id="t",
        steps=(
            Step(id="a", agent="planner", instruction="x", reads=("b",)),
            Step(id="b", agent="reporter", instruction="y"),
        ),
    )
    with pytest.raises(PlanInvalid):
        validate_plan(plan, AGENT_KEYS, ROOT_ARTIFACTS)


def test_self_read_is_rejected() -> None:
    plan = Plan(
        task_id="t",
        steps=(Step(id="a", agent="planner", instruction="x", reads=("a",)),),
    )
    with pytest.raises(PlanInvalid):
        validate_plan(plan, AGENT_KEYS, ROOT_ARTIFACTS)


def test_empty_instruction_is_rejected() -> None:
    plan = Plan(
        task_id="t",
        steps=(Step(id="a", agent="planner", instruction="   "),),
    )
    with pytest.raises(PlanInvalid) as caught:
        validate_plan(plan, AGENT_KEYS, ROOT_ARTIFACTS)
    assert "instruction is empty" in str(caught.value)


def test_malformed_step_id_is_rejected() -> None:
    plan = Plan(
        task_id="t",
        steps=(Step(id="Step One", agent="planner", instruction="x"),),
    )
    with pytest.raises(PlanInvalid) as caught:
        validate_plan(plan, AGENT_KEYS, ROOT_ARTIFACTS)
    assert "id must match" in str(caught.value)


def test_step_count_above_max_is_rejected() -> None:
    plan = Plan(
        task_id="t",
        steps=(
            Step(id="a", agent="planner", instruction="x"),
            Step(id="b", agent="reporter", instruction="y"),
        ),
        max_steps=1,
    )
    with pytest.raises(PlanInvalid) as caught:
        validate_plan(plan, AGENT_KEYS, ROOT_ARTIFACTS)
    assert "max_steps" in str(caught.value)


def test_every_problem_is_reported_at_once() -> None:
    plan = Plan(
        task_id="t",
        steps=(
            Step(id="a", agent="nobody", instruction=""),
            Step(id="a", agent="nobody", instruction="x", reads=("ghost",)),
        ),
    )
    with pytest.raises(PlanInvalid) as caught:
        validate_plan(plan, AGENT_KEYS, ROOT_ARTIFACTS)
    assert len(caught.value.problems) >= 4


def test_custom_root_artifacts_are_readable() -> None:
    plan = Plan(
        task_id="t",
        steps=(Step(id="a", agent="planner", instruction="x", reads=("schema",)),),
    )
    validate_plan(plan, AGENT_KEYS, ("schema",))


# --------------------------------------------------------------------------
# Prefix runs (--plan)
# --------------------------------------------------------------------------


def test_select_steps_keeps_the_requested_step() -> None:
    prefix = select_steps(build_plan("tsk_1"), ("plan",))
    assert [step.id for step in prefix.steps] == ["plan"]
    assert prefix.task_id == "tsk_1"
    validate_plan(prefix, AGENT_KEYS, ROOT_ARTIFACTS)


def test_select_steps_preserves_workflow_order() -> None:
    prefix = select_steps(build_plan("tsk_1"), ("investigate", "plan"))
    assert [step.id for step in prefix.steps] == ["plan", "investigate"]


def test_select_steps_with_no_match_is_rejected() -> None:
    with pytest.raises(PlanInvalid) as caught:
        select_steps(build_plan("tsk_1"), ("ghost",))
    assert "ghost" in str(caught.value)


def test_select_steps_does_not_mutate_the_original_plan() -> None:
    plan = build_plan("tsk_1")
    select_steps(plan, ("plan",))
    assert len(plan.steps) == 3
