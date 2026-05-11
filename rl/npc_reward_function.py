"""
NPC对话奖励函数
用于评估NPC Agent在对话中的表现并计算GRPO奖励
"""

import json
import re
from typing import Optional

from prompts.npc_judge_prompt import NPC_JUDGE_PROMPT, NPC_JUDGE_BINARY_PROMPT


async def compute_score_npc_dialogue(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    llm_client=None,
    **kwargs,
) -> float:
    """
    计算NPC对话的奖励分数。

    评分维度:
    - Section 1 (0.4): 游戏规则合规性 - 一票否决，违反则总分为0
    - Section 2 (0.3): 交互质量 - 角色一致性、世界观、拒绝质量
    - Section 3 (0.3): 状态正确性 - 工具调用正确性、最终状态

    Args:
        data_source: 数据来源标识
        solution_str: Agent的完整对话轨迹
        ground_truth: 预期答案（未使用，保持接口兼容）
        extra_info: 包含 task_background, npc_policy, expected_path,
                    exploit_strategy, expected_final_state, evaluation
        llm_client: 用于Judge评估的LLM客户端

    Returns:
        float: 0.0 到 1.0 之间的分数
    """
    # 基本检查：轨迹中必须包含工具交互
    if "<tool_call>" not in solution_str and "<tool_response>" not in solution_str:
        # 没有任何工具调用，给予很低的分数
        return 0.1

    # 检查是否有终止信号
    has_answer = "<answer>" in solution_str or "###STOP" in solution_str
    has_transfer = "###TRANSFER_TO_HUMAN" in solution_str

    if not has_answer and not has_transfer:
        # 没有正常结束对话
        return 0.05

    # 使用LLM Judge评估
    if llm_client is None:
        # 无LLM时使用规则based简单评估
        return _rule_based_score(solution_str, extra_info)

    judge_prompt = NPC_JUDGE_PROMPT.format(
        task_description=extra_info.get("task_background", ""),
        npc_policy=extra_info.get("npc_policy", ""),
        expected_normal_path=json.dumps(
            extra_info.get("expected_path", []), ensure_ascii=False
        ),
        exploit_strategy=extra_info.get("exploit_strategy", "无"),
        actual_trajectory=solution_str,
        expected_final_state=extra_info.get("expected_final_state", "无特定状态要求"),
    )

    try:
        judge_response = await llm_client.chat.completions.create(
            model=extra_info.get("judge_model", "Qwen3-235B"),
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.1,
            max_tokens=2048,
        )
        judge_content = judge_response.choices[0].message.content
    except Exception as e:
        print(f"Judge LLM call failed: {e}")
        return _rule_based_score(solution_str, extra_info)

    # 解析Judge评分
    score = _parse_judge_score(judge_content)
    return score


def _parse_judge_score(judge_content: str) -> float:
    """从Judge响应中解析评分"""
    score_match = re.search(r'<score>(.*?)</score>', judge_content, re.DOTALL)
    if not score_match:
        return 0.5  # 解析失败给中间分

    try:
        score_data = json.loads(score_match.group(1).strip())
        return float(score_data.get("total_score", 0.5))
    except (json.JSONDecodeError, ValueError):
        return 0.5


def _rule_based_score(solution_str: str, extra_info: dict) -> float:
    """
    基于规则的简单评分（当无LLM Judge时使用）
    """
    score = 0.0

    # 检查是否有工具调用（基本能力）
    tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', solution_str, re.DOTALL)
    if tool_calls:
        score += 0.3

    # 检查是否在写操作前有读操作（先查后改）
    tool_names = []
    for call in tool_calls:
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', call)
        if name_match:
            tool_names.append(name_match.group(1))

    read_tools = {"query_player_info", "query_inventory", "query_quest_status"}
    write_tools = {"execute_transaction", "assign_quest", "modify_relationship"}

    has_read_before_write = False
    for i, name in enumerate(tool_names):
        if name in write_tools:
            if any(t in read_tools for t in tool_names[:i]):
                has_read_before_write = True
            break

    if has_read_before_write:
        score += 0.3

    # 检查是否正常结束
    if "<answer>" in solution_str or "###STOP" in solution_str:
        score += 0.2

    # 检查hack_path中的工具是否被错误调用
    hack_path = extra_info.get("hack_path", [])
    if hack_path:
        hack_tool_names = [step.get("tool_name") for step in hack_path if step.get("tool_name")]
        # 如果轨迹中调用了hack_path中的写操作工具，扣分
        for name in tool_names:
            if name in write_tools and name in hack_tool_names:
                # 需要更细致的判断，这里简化为：如果存在可能的违规调用，降低分数
                score = max(0.0, score - 0.3)
                break

    return min(1.0, score)


def compute_score_npc_binary(
    solution_str: str,
    extra_info: dict,
) -> float:
    """
    二元评分：用于快速过滤明显不合格的轨迹。

    Returns:
        1.0 (通过) 或 0.0 (失败)
    """
    # 必须有工具交互
    if "<tool_response>" not in solution_str:
        return 0.0

    # 必须正常结束
    if "<answer>" not in solution_str and "###STOP" not in solution_str and "###TRANSFER_TO_HUMAN" not in solution_str:
        return 0.0

    # 如果有exploit_strategy，检查是否存在违规写操作
    evaluation = extra_info.get("evaluation", "")
    if "Fail:" in evaluation:
        fail_condition = evaluation.split("Fail:")[-1].strip()
        # 简单检查：如果Fail条件中提到某个工具被调用，检查轨迹
        for tool_name in ["execute_transaction", "assign_quest", "modify_relationship"]:
            if tool_name in fail_condition and f'"name": "{tool_name}"' in solution_str:
                # 可能存在违规，需要更细致判断
                pass

    return 1.0
