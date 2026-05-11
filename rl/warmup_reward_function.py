"""
游戏推理Warm-up奖励函数
二元奖励：工具调用正确且结论正确 = 1.0，否则 = 0.0
不使用LLM Judge（纯规则判定，快速且确定性）
"""

import json
import re
from typing import List, Dict, Any


async def compute_score_warmup(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    **kwargs,
) -> float:
    """
    Warm-up阶段的二元奖励函数。

    评分规则：
    1. 必须包含至少一个tool_call (0.3)
    2. 第一个tool_call必须是查询操作 (0.2)
    3. 如果should_succeed=True，必须调用了写操作 (0.25)
       如果should_succeed=False，不能调用写操作 (0.25)
    4. 必须有answer/终止信号 (0.25)

    总分 = 各项得分之和，最终二值化：≥0.7 → 1.0, <0.7 → 0.0
    """
    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except (json.JSONDecodeError, TypeError):
        gt = {}

    should_succeed = gt.get("should_succeed", True)
    correct_tool_calls = gt.get("correct_tool_calls", [])

    score = 0.0

    # ---- Rule 1: 包含tool_call ----
    tool_calls_found = re.findall(r'<tool_call>(.*?)</tool_call>', solution_str, re.DOTALL)
    if tool_calls_found:
        score += 0.3

    # ---- Rule 2: 第一个tool_call是查询操作 ----
    read_tools = {"query_player_info", "query_inventory", "query_quest_status"}
    write_tools = {"execute_transaction", "assign_quest", "modify_relationship"}

    called_tool_names = []
    for tc in tool_calls_found:
        try:
            tc_data = json.loads(tc.strip())
            name = tc_data.get("name", "")
            called_tool_names.append(name)
        except json.JSONDecodeError:
            pass

    if called_tool_names and called_tool_names[0] in read_tools:
        score += 0.2

    # ---- Rule 3: 写操作正确性 ----
    has_write_op = any(n in write_tools for n in called_tool_names)

    if should_succeed and has_write_op:
        score += 0.25  # 应该执行且确实执行了
    elif not should_succeed and not has_write_op:
        score += 0.25  # 应该拒绝且确实没执行
    # 其他情况不得分

    # ---- Rule 4: 正常终止 ----
    has_termination = (
        "<answer>" in solution_str
        or "###STOP" in solution_str
        or "###TRANSFER_TO_HUMAN" in solution_str
    )
    if has_termination:
        score += 0.25

    # 二值化
    return 1.0 if score >= 0.7 else 0.0
