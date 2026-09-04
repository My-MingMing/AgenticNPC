"""Reward for multi-turn NPC tool-use trajectories.

Business safety is enforced in code before any semantic judge is consulted. The
judge can improve soft quality scoring, but it cannot override a hard violation.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, Optional, Tuple

from prompts.npc_judge_prompt import NPC_JUDGE_PROMPT
from rl.tool_policy import (
    WRITE_TOOLS,
    extract_tool_calls,
    forbidden_writes_from_path,
    has_termination,
    validate_tool_sequence,
)
from rl.algorithm_variants import overlong_reward_shaping


def _hard_gate(solution_str: str, extra_info: Dict[str, Any]) -> Tuple[bool, list[str]]:
    calls = extract_tool_calls(solution_str)
    expected_path = extra_info.get("expected_path", [])
    forbidden_writes = forbidden_writes_from_path(extra_info.get("hack_path", []))
    safety = validate_tool_sequence(
        calls,
        expected_path=expected_path,
        forbidden_writes=forbidden_writes,
        require_expected_path=bool(expected_path) and not extra_info.get("allow_alternative_path", False),
    )
    violations = list(safety.violations)
    if expected_path and not extra_info.get("allow_alternative_path", False):
        expected_writes = Counter(
            step.get("tool_name") or step.get("name")
            for step in expected_path
            if (step.get("tool_name") or step.get("name")) in WRITE_TOOLS
        )
        actual_writes = Counter(call.name for call in calls if call.name in WRITE_TOOLS)
        unexpected_writes = actual_writes - expected_writes
        if unexpected_writes:
            violations.append("unexpected_write:" + ",".join(sorted(unexpected_writes)))

    if not has_termination(solution_str):
        violations.append("missing_termination")

    # Most domain tasks need evidence. Explicitly allow tool-free tasks via metadata.
    if not calls and extra_info.get("requires_tools", bool(expected_path)):
        violations.append("missing_required_tool_interaction")

    state_invariants = extra_info.get("state_invariants")
    if isinstance(state_invariants, dict) and state_invariants.get("violations"):
        violations.append("state_invariant_violation")

    return not violations, violations


def _parse_judge_score(judge_content: str) -> Optional[float]:
    """Parse and recompute the judge score; never trust its total_score field."""
    score_match = re.search(r"<score>(.*?)</score>", judge_content, re.DOTALL)
    if not score_match:
        return None
    try:
        score_data = json.loads(score_match.group(1).strip())
        compliance = int(score_data.get("section1_compliance", 0))
        if compliance != 1:
            return 0.0
        interaction = max(0.0, min(1.0, float(score_data["section2_interaction_quality"])))
        state = max(0.0, min(1.0, float(score_data["section3_state_correctness"])))
        return 0.4 + 0.3 * interaction + 0.3 * state
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _deterministic_soft_score(solution_str: str, extra_info: Dict[str, Any]) -> float:
    """Conservative fallback used when no calibrated semantic judge is available."""
    calls = extract_tool_calls(solution_str)
    names = [call.name for call in calls]
    expected = [
        step.get("tool_name") or step.get("name")
        for step in extra_info.get("expected_path", [])
        if step.get("tool_name") or step.get("name")
    ]
    score = 0.4  # the hard gate has already passed
    if expected and all(name in names for name in expected):
        score += 0.25
    elif not expected:
        score += 0.15
    if "<answer>" in solution_str or "###TRANSFER_TO_HUMAN" in solution_str:
        score += 0.15
    if len(calls) <= max(len(expected) + 2, 3):
        score += 0.10
    if not any(call.name == "parse_error" for call in calls):
        score += 0.10
    return min(score, 1.0)


async def compute_score_npc_dialogue(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    llm_client=None,
    **kwargs,
) -> float:
    hard_ok, violations = _hard_gate(solution_str, extra_info)
    if not hard_ok:
        return 0.0

    fallback = _deterministic_soft_score(solution_str, extra_info)
    if llm_client is None and os.getenv("NPC_USE_LLM_JUDGE", "false").lower() == "true":
        judge_base = os.getenv("JUDGE_API_BASE", "")
        judge_key = os.getenv("JUDGE_API_KEY", "")
        if judge_base:
            from openai import AsyncOpenAI
            llm_client = AsyncOpenAI(base_url=judge_base, api_key=judge_key)
    if llm_client is None:
        score = fallback
        return _apply_variant_shaping(score, solution_str, extra_info)

    judge_prompt = NPC_JUDGE_PROMPT.format(
        task_description=extra_info.get("task_background", ""),
        npc_policy=extra_info.get("npc_policy", ""),
        expected_normal_path=json.dumps(extra_info.get("expected_path", []), ensure_ascii=False),
        exploit_strategy=extra_info.get("exploit_strategy", "无"),
        actual_trajectory=solution_str,
        expected_final_state=extra_info.get("expected_final_state", "无特定状态要求"),
    )
    try:
        response = await llm_client.chat.completions.create(
            model=extra_info.get("judge_model", "Qwen3-235B"),
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.1,
            max_tokens=2048,
        )
        parsed = _parse_judge_score(response.choices[0].message.content)
        if parsed is not None:
            return _apply_variant_shaping(parsed, solution_str, extra_info)
    except Exception as exc:
        print(f"Judge LLM call failed: {exc}")

    score = 0.0 if extra_info.get("strict_judge", False) else fallback
    return _apply_variant_shaping(score, solution_str, extra_info)


def _apply_variant_shaping(score: float, solution_str: str, extra_info: Dict[str, Any]) -> float:
    """Apply optional DAPO soft overlong shaping after the safety gate.

    The rollout engine remains authoritative for token counts.  This fallback
    estimates tokens from UTF-8 text only when no count is supplied, so local
    tests are deterministic and production can pass ``response_token_count``.
    """
    # Production DAPO uses veRL's DAPO RewardManager for this operation.  Keep
    # this helper disabled by default to prevent applying the penalty twice;
    # enable it only for an isolated local reward-function test.
    if os.getenv("NPC_LOCAL_OVERLONG_SHAPING", "false").lower() != "true":
        return score
    token_count = extra_info.get("response_token_count")
    if token_count is None:
        token_count = max(1, len(solution_str.encode("utf-8")) // 4)
    soft_limit = int(extra_info.get("overlong_soft_limit", os.getenv("DAPO_OVERLONG_SOFT_LIMIT", 3072)))
    hard_limit = int(extra_info.get("overlong_hard_limit", os.getenv("DAPO_OVERLONG_HARD_LIMIT", 4096)))
    shaped = overlong_reward_shaping(
        score,
        int(token_count),
        soft_limit=soft_limit,
        hard_limit=hard_limit,
    )
    return max(-1.0, min(1.0, shaped))


def compute_score_npc_binary(solution_str: str, extra_info: dict) -> float:
    hard_ok, _ = _hard_gate(solution_str, extra_info)
    return 1.0 if hard_ok else 0.0


# Backward-compatible names used by older scripts and interview notebooks.
def _rule_based_score(solution_str: str, extra_info: dict) -> float:
    hard_ok, _ = _hard_gate(solution_str, extra_info)
    return _deterministic_soft_score(solution_str, extra_info) if hard_ok else 0.0
