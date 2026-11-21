"""How agents read structured replies from a model: one convention, shared.

A structured reply is one JSON object, possibly wrapped in prose or code fences.
Each agent parses it against its own contract; what is shared is the reading, the
failure type, and the wording of the one repair turn, so a model gets consistent
feedback no matter which agent rejected it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any


class ReplyInvalid(Exception):
    """A reply violated an agent's contract. Carries every problem found."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems: tuple[str, ...] = tuple(problems)
        super().__init__("; ".join(self.problems))


def extract_json(text: str) -> Any:
    """Find the JSON object in a reply that may be wrapped in prose or fences.

    Slicing from the first `{` to the last `}` is deliberate: it tolerates the
    chatter models add, and anything it cannot parse is a typed rejection the
    agent can send back rather than an exception that escapes the run.
    """
    candidate = (text or "").strip()
    if not candidate:
        raise ReplyInvalid(["the reply was empty"])

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < start:
        raise ReplyInvalid(["the reply does not contain a JSON object"])

    try:
        return json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ReplyInvalid(
            [f"the reply is not valid JSON: {exc.msg} (line {exc.lineno})"]
        ) from None


def repair_instruction(
    problems: Sequence[str],
    *,
    validator: str,
    valid_ids: str = "",
    allowed_values: str = "",
) -> str:
    """The user message that sends a rejected reply back with its problems."""
    listed = "\n".join(f"- {problem}" for problem in problems)
    extras = ""
    if valid_ids:
        extras += f"\n- Valid query_id values: {valid_ids}"
    if allowed_values:
        extras += f"\n- Allowed values: {allowed_values}"
    return (
        f"Your previous reply was rejected by the {validator}:\n"
        f"{listed}"
        f"{extras}\n\n"
        "Rules you must follow:\n"
        "- Every number in a finding must appear EXACTLY as-is in the query "
        "results above. Do not compute, subtract, round, or derive any new number.\n"
        "- If a comparison needs a number that is not in the results, describe the "
        "relationship in prose instead of inventing a figure.\n"
        "- Remove any number key whose value does not come directly from a cited "
        "query row.\n\n"
        "Return the corrected JSON object only. Keep the parts that were already "
        "correct."
    )
