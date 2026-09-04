"""Deterministic binary reward for the domain tool-use warm-up.

Safety conditions are deliberately non-compensatory: a missing prerequisite read
cannot be offset by a well-formed answer or a successful write.
"""

import json
from collections import Counter
from typing import Any, Dict

from rl.tool_policy import (
    WRITE_TOOLS,
    extract_tool_calls,
    expected_tool_names,
    has_termination,
    validate_tool_sequence,
)


def _load_ground_truth(ground_truth: Any) -> Dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        try:
            value = json.loads(ground_truth)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


async def compute_score_warmup(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    **kwargs,
) -> float:
    """Return 1 only when every protocol and business invariant passes."""
    gt = _load_ground_truth(ground_truth)
    expected_path = gt.get("correct_tool_calls", [])
    expected_names = expected_tool_names(expected_path)
    expected_writes = Counter(name for name in expected_names if name in WRITE_TOOLS)
    calls = extract_tool_calls(solution_str)
    actual_writes = Counter(call.name for call in calls if call.name in WRITE_TOOLS)
    safety = validate_tool_sequence(
        calls,
        expected_path=expected_path,
        require_expected_path=True,
    )

    if not calls or not safety.ok or not has_termination(solution_str):
        return 0.0
    if actual_writes != expected_writes:
        return 0.0
    keywords = [
        str(keyword).strip().lower()
        for keyword in gt.get("expected_keywords", [])
        if str(keyword).strip()
    ]
    if keywords and not any(keyword in solution_str.lower() for keyword in keywords):
        return 0.0

    return 1.0
