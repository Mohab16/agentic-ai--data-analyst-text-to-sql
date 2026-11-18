"""Golden evaluation cases: the full pipeline with scripted models.

Each case runs the complete plan → investigate → answer pipeline with a
ScriptedLLM and asserts that the final report's numbers are consistent with
the query results that produced them. These are the acceptance criteria for
Phase 8: golden evaluations, README completion, and dependency audit.

The test harness uses the `scripted_run` helper from `test_main.py`, which
wires a scripted model in place of the provider while keeping the tools real.
"""

from __future__ import annotations

import json

import pytest

from app.main import EXIT_OK, EXIT_RUN_FAILED, main

from tests.test_main import (
    ANSWER_PAYLOAD,
    ANSWER_PAYLOAD_6273,
    PLAN_PAYLOAD,
    call_execute,
    findings_reply,
    scripted_run,
)

# A reporter reply for an investigation with no findings: confidence must be "low"
# (reporter._check_confidence rejects "medium"/"high" when the investigation
# produced zero findings), and at least one limitation is required.
_UNANSWERABLE_ANSWER = {
    "headline": "The question cannot be answered from this data.",
    "answer": (
        "The schema does not contain revenue figures, so this question "
        "cannot be answered from the available tables."
    ),
    "findings": [],
    "assumptions": [],
    "limitations": ["no revenue or financial columns exist in the schema"],
    "confidence": "low",
}

QUESTION = "Why did our revenue drop this month?"
COUNT_SQL = "SELECT COUNT(*) AS completed_orders FROM orders WHERE status = 'completed'"


# --------------------------------------------------------------------------- #
# Case 1: a truthful question that can be answered from the data
# --------------------------------------------------------------------------- #

def test_golden_truthful_question_matches_the_query(
    configured: None,
    monkeypatch: pytest.MonkeyPatch,
    scripted_llm,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Phase 8: the report's numbers must match the rows behind them."""
    scripted_run(
        monkeypatch,
        scripted_llm,
        json.dumps(PLAN_PAYLOAD),
        call_execute(COUNT_SQL),
        findings_reply(numbers={"orders": 2797}),
        reporter_replies=(json.dumps(ANSWER_PAYLOAD),),
    )

    assert main([QUESTION]) == EXIT_OK
    out = capsys.readouterr().out
    assert "findings: 1" in out
    assert "q_001" in out
    assert COUNT_SQL in out
    assert "stop reason: model_finished" in out
    assert "answer" in out
    assert "Completed orders rose." in out
    assert "confidence: medium" in out


# --------------------------------------------------------------------------- #
# Case 2: an unanswerable question (model refuses with open questions)
# --------------------------------------------------------------------------- #

def test_golden_unanswerable_question_is_reported_as_such(
    configured: None,
    monkeypatch: pytest.MonkeyPatch,
    scripted_llm,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Phase 8: the pipeline surfaces unanswerable questions without fabricating."""
    scripted_run(
        monkeypatch,
        scripted_llm,
        json.dumps(PLAN_PAYLOAD),
        call_execute(COUNT_SQL),
        json.dumps(
            {
                "findings": [],
                "open_questions": ["the schema does not contain revenue figures"],
                "notes": ["need a financial table to answer"],
            }
        ),
        reporter_replies=(json.dumps(_UNANSWERABLE_ANSWER),),
    )

    assert main([QUESTION]) == EXIT_OK
    out = capsys.readouterr().out
    assert "open questions:" in out
    assert "revenue figures" in out


# --------------------------------------------------------------------------- #
# Case 3: a finding whose evidence query never ran is rejected
# --------------------------------------------------------------------------- #

def test_golden_unbacked_claim_is_rejected(
    configured: None,
    monkeypatch: pytest.MonkeyPatch,
    scripted_llm,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Phase 8: a claim citing a query that never ran is rejected, then fails."""
    scripted_run(
        monkeypatch,
        scripted_llm,
        json.dumps(PLAN_PAYLOAD),
        call_execute(COUNT_SQL),
        findings_reply(evidence="q_999"),
        findings_reply(evidence="q_999"),
    )

    assert main([QUESTION]) == EXIT_RUN_FAILED
    assert "q_999" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Case 4: a mismatch between the finding number and the query result is caught
# --------------------------------------------------------------------------- #

def test_golden_number_mismatch_is_rejected(
    configured: None,
    monkeypatch: pytest.MonkeyPatch,
    scripted_llm,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Phase 8: the pipeline rejects a finding whose number contradicts the rows."""
    scripted_run(
        monkeypatch,
        scripted_llm,
        json.dumps(PLAN_PAYLOAD),
        call_execute(COUNT_SQL),
        findings_reply(numbers={"orders": 9999}),
        findings_reply(numbers={"orders": 9999}),
    )

    assert main([QUESTION]) == EXIT_RUN_FAILED
    err = capsys.readouterr().err
    assert "9999" in err
    assert "2797" in err


# --------------------------------------------------------------------------- #
# Case 5: the full JSON payload carries the evidence end to end
# --------------------------------------------------------------------------- #

def test_golden_json_payload_carries_evidence(
    configured: None,
    monkeypatch: pytest.MonkeyPatch,
    scripted_llm,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Phase 8: the JSON output carries the evidence through every step."""
    scripted_run(
        monkeypatch,
        scripted_llm,
        json.dumps(PLAN_PAYLOAD),
        call_execute(COUNT_SQL),
        findings_reply(),
        reporter_replies=(json.dumps(ANSWER_PAYLOAD),),
    )

    assert main(["--json", QUESTION]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["investigation"]["findings"][0]["numbers"] == {"orders": 2797.0}
    queries = payload["investigation"]["queries"]
    assert len(queries) == 1
    assert queries[0]["status"] == "ok"
    assert queries[0]["id"] == "q_001"
    assert queries[0]["row_count"] == 1
    assert "completed_orders" in queries[0]["preview"]
    assert payload["tokens_in"] > 0
    assert payload["answer"]["headline"] == "Completed orders rose."