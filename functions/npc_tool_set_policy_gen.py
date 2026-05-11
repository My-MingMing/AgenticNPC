"""
NPC行为树和工具集生成函数
根据NPC角色种子生成游戏专用的行为树策略
"""

import json
import re
from typing import Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configuration import ModelConfiguration

# Load game tools library
GAME_TOOLS_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'game_tools_library.json')
with open(GAME_TOOLS_PATH, 'r', encoding='utf-8') as f:
    GAME_TOOLS_LIBRARY = json.load(f)


def _build_npc_prompt(npc_persona: str, normal_workflow: Optional[str],
                      game_context: dict, npc_type: str) -> str:
    """构建NPC行为树生成的完整prompt"""

    from prompts.npc_system_prompt import NPC_BEHAVIOR_TREE_PROMPT

    game_tools_str = json.dumps(GAME_TOOLS_LIBRARY["tools"], ensure_ascii=False, indent=2)

    # 根据NPC类型选择相关工具子集提示
    relevant_tools_hint = ""
    if npc_type == "merchant":
        relevant_tools_hint = "该NPC主要使用: query_player_info, query_inventory, check_item_requirements, calculate_price, execute_transaction, execute_buyback, modify_relationship"
    elif npc_type == "quest_giver":
        relevant_tools_hint = "该NPC主要使用: query_player_info, query_player_achievements, query_quest_status, assign_quest, submit_quest, modify_relationship"
    elif npc_type == "guard":
        relevant_tools_hint = "该NPC主要使用: query_player_info, query_inventory (检查通行证/钥匙), query_world_state, query_npc_schedule"
    elif npc_type == "information":
        relevant_tools_hint = "该NPC主要使用: query_player_info, query_player_achievements, modify_relationship, query_world_state, query_mail"
    elif npc_type == "service":
        relevant_tools_hint = "该NPC主要使用: query_player_info, query_inventory, query_player_skills, execute_repair, execute_enchant, teach_skill, calculate_price"

    context_str = json.dumps(game_context, ensure_ascii=False, indent=2) if game_context else "无额外上下文"

    prompt = NPC_BEHAVIOR_TREE_PROMPT.format(
        npc_persona=f"类型: {npc_type}\n角色描述: {npc_persona}\n游戏上下文: {context_str}\n工具提示: {relevant_tools_hint}",
        normal_workflow=normal_workflow or "未提供，请根据NPC类型设计合理的工作流",
        game_tools=game_tools_str,
    )

    return prompt


def _parse_response(response: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """从LLM响应中解析task、tools(behavior_tree中的工具使用)、policy"""

    task_match = re.search(r'<task>(.*?)</task>', response, re.DOTALL)
    policy_match = re.search(r'<npc_policy>(.*?)</npc_policy>', response, re.DOTALL)
    tree_match = re.search(r'<behavior_tree>(.*?)</behavior_tree>', response, re.DOTALL)

    task = task_match.group(1).strip() if task_match else None
    policy = tree_match.group(1).strip() if tree_match else None
    npc_policy_doc = policy_match.group(1).strip() if policy_match else None

    return task, npc_policy_doc, policy


def generate_npc_behavior_tree(
    cfg: ModelConfiguration,
    npc_persona: str,
    normal_workflow: Optional[str],
    game_context: dict,
    npc_type: str,
) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    生成NPC的行为树和策略。

    Returns:
        (all_content, task, tools_json_str, policy_tree_json_str)
    """
    from openai import OpenAI

    prompt = _build_npc_prompt(npc_persona, normal_workflow, game_context, npc_type)

    client = OpenAI(base_url=cfg.api_base, api_key=cfg.api_key)

    try:
        response = client.chat.completions.create(
            model=cfg.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        all_content = response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return "", None, None, None

    task, npc_policy_doc, policy_tree = _parse_response(all_content)

    # 工具使用原始game_tools_library中的定义
    tools_str = json.dumps(GAME_TOOLS_LIBRARY["tools"], ensure_ascii=False)

    if task is None or policy_tree is None:
        print(f"Warning: Failed to parse behavior tree from response")
        return all_content, None, None, None

    return all_content, task, tools_str, policy_tree
