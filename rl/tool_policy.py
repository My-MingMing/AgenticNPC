"""Shared tool parsing and safety policy for training, evaluation, and serving."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set


READ_TOOLS: Set[str] = {
    "query_player_info",
    "query_player_skills",
    "query_player_achievements",
    "query_inventory",
    "check_item_requirements",
    "calculate_price",
    "query_quest_status",
    "query_mail",
    "query_world_state",
    "query_npc_schedule",
}

WRITE_TOOLS: Set[str] = {
    "execute_transaction",
    "execute_buyback",
    "assign_quest",
    "submit_quest",
    "modify_relationship",
    "execute_repair",
    "execute_enchant",
    "teach_skill",
}

# Generic minimums. A scenario's expected path may impose stricter requirements.
WRITE_EVIDENCE: Mapping[str, Set[str]] = {
    "execute_transaction": {
        "query_player_info",
        "query_inventory",
        "check_item_requirements",
    },
    "execute_buyback": {"query_inventory"},
    "assign_quest": {"query_quest_status"},
    "submit_quest": {"query_quest_status"},
    "modify_relationship": {"query_player_info"},
    "execute_repair": {"query_inventory"},
    "execute_enchant": {"query_inventory"},
    "teach_skill": {"query_player_skills"},
}

# Entity keys that bind evidence to the command it authorizes. Keys absent from
# the write are ignored for backward-compatible tool-free unit fixtures; when a
# write carries one of these keys, a prior read must carry the same value.
EVIDENCE_SCOPE_KEYS: Mapping[tuple[str, str], Sequence[tuple[str, str]]] = {
    ("execute_transaction", "query_player_info"): (("player_id", "player_id"),),
    # Inventory queries identify an owner/category rather than one item; the
    # scenario path binds the exact inventory query when it is authoritative.
    ("execute_transaction", "query_inventory"): (),
    ("execute_transaction", "check_item_requirements"): (
        ("player_id", "player_id"),
        ("item_id", "item_id"),
    ),
    ("execute_buyback", "query_inventory"): (),
    ("assign_quest", "query_quest_status"): (
        ("player_id", "player_id"),
        ("quest_id", "quest_id"),
    ),
    ("submit_quest", "query_quest_status"): (
        ("player_id", "player_id"),
        ("quest_id", "quest_id"),
    ),
    ("modify_relationship", "query_player_info"): (("player_id", "player_id"),),
    ("execute_repair", "query_inventory"): (("owner_id", "player_id"),),
    ("execute_enchant", "query_inventory"): (("owner_id", "player_id"),),
    ("teach_skill", "query_player_skills"): (("player_id", "player_id"),),
}

TERMINATION_MARKERS = ("<answer>", "###STOP", "###TRANSFER_TO_HUMAN")


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    valid: bool = True


@dataclass
class SafetyCheck:
    ok: bool
    violations: List[str]
    calls: List[ToolCall]


def _decode_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def normalize_tool_call(data: Any, raw: str = "") -> ToolCall:
    """Normalize direct calls and the veRL ``mock_tool(query=...)`` envelope."""
    data = _decode_json(data)
    if not isinstance(data, dict):
        return ToolCall(name="parse_error", raw=raw or str(data), valid=False)

    if isinstance(data.get("function"), dict):
        fn = data["function"]
        data = {"name": fn.get("name"), "arguments": fn.get("arguments", {})}

    name = data.get("name") or data.get("tool_name")
    args = data.get("arguments", data.get("parameters", {}))
    args = _decode_json(args)
    if not isinstance(args, dict):
        args = {"raw": args}

    if name == "mock_tool":
        inner = args.get("query", data.get("query"))
        inner = _decode_json(inner)
        if isinstance(inner, dict):
            return normalize_tool_call(inner, raw=raw)
        return ToolCall(name="parse_error", arguments=args, raw=raw, valid=False)

    if not isinstance(name, str) or not name:
        return ToolCall(name="parse_error", arguments=args, raw=raw, valid=False)
    return ToolCall(name=name, arguments=args, raw=raw, valid=True)


def extract_tool_calls(text: str) -> List[ToolCall]:
    calls: List[ToolCall] = []
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        raw = match.group(1).strip()
        try:
            calls.append(normalize_tool_call(json.loads(raw), raw=raw))
        except json.JSONDecodeError:
            calls.append(ToolCall(name="parse_error", raw=raw, valid=False))
    return calls


def expected_tool_names(path: Optional[Iterable[Mapping[str, Any]]]) -> List[str]:
    names: List[str] = []
    for step in path or []:
        name = step.get("tool_name") or step.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def is_subsequence(expected: Sequence[str], actual: Sequence[str]) -> bool:
    if not expected:
        return True
    cursor = 0
    for name in actual:
        if name == expected[cursor]:
            cursor += 1
            if cursor == len(expected):
                return True
    return False


def _arguments_include(actual: Mapping[str, Any], expected: Any) -> bool:
    """Return true when all explicitly expected arguments match."""
    if not isinstance(expected, Mapping):
        return True
    return all(actual.get(key) == value for key, value in expected.items())


def is_call_subsequence(
    expected_path: Optional[Iterable[Mapping[str, Any]]],
    calls: Sequence[ToolCall],
) -> bool:
    """Match ordered tool names and any Oracle-provided argument constraints."""
    expected = list(expected_path or [])
    if not expected:
        return True
    cursor = 0
    for call in calls:
        step = expected[cursor]
        step_name = step.get("tool_name") or step.get("name")
        step_args = step.get("input", step.get("arguments"))
        if call.name == step_name and _arguments_include(call.arguments, step_args):
            cursor += 1
            if cursor == len(expected):
                return True
    return False


def _has_scoped_evidence(read: ToolCall, write: ToolCall) -> bool:
    key_pairs = EVIDENCE_SCOPE_KEYS.get((write.name, read.name), ())
    for read_key, write_key in key_pairs:
        if write_key in write.arguments and read.arguments.get(read_key) != write.arguments[write_key]:
            return False
    return True


def validate_tool_sequence(
    calls: Sequence[ToolCall],
    *,
    expected_path: Optional[Iterable[Mapping[str, Any]]] = None,
    forbidden_writes: Optional[Iterable[str]] = None,
    require_expected_path: bool = False,
) -> SafetyCheck:
    violations: List[str] = []
    names = [call.name for call in calls]

    if any(not call.valid for call in calls):
        violations.append("invalid_tool_call_json")

    expected = expected_tool_names(expected_path)
    if require_expected_path and not is_call_subsequence(expected_path, calls):
        violations.append("missing_or_misordered_expected_tools")

    forbidden = set(forbidden_writes or []) & WRITE_TOOLS
    used_forbidden = forbidden.intersection(names)
    if used_forbidden:
        violations.append("forbidden_write:" + ",".join(sorted(used_forbidden)))

    # A write must be supported by evidence observed earlier in the trajectory.
    for index, name in enumerate(names):
        if name not in WRITE_TOOLS:
            continue
        prior_calls = calls[:index]
        required = set(WRITE_EVIDENCE.get(name, set()))
        # Scenario paths are authoritative when they explicitly place reads before this write.
        if name in expected:
            write_index = expected.index(name)
            required.update(t for t in expected[:write_index] if t in READ_TOOLS)
        missing = {
            evidence_name
            for evidence_name in required
            if not any(
                read.name == evidence_name and _has_scoped_evidence(read, calls[index])
                for read in prior_calls
            )
        }
        if missing:
            violations.append(f"missing_evidence_for_{name}:" + ",".join(sorted(missing)))

    return SafetyCheck(ok=not violations, violations=violations, calls=list(calls))


def has_termination(text: str) -> bool:
    return any(marker in text for marker in TERMINATION_MARKERS)


def forbidden_writes_from_path(path: Optional[Iterable[Mapping[str, Any]]]) -> Set[str]:
    return set(expected_tool_names(path)) & WRITE_TOOLS
