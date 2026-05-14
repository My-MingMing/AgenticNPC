"""
NPC对话数据合成 - LangGraph工作流定义
三阶段流程：行为树生成 → 策略+测试用例 → 用例精炼
"""

import os
import json
import yaml
import threading
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configuration import ModelConfiguration


log_file_lock = threading.Lock()


class NPCDataGenState(TypedDict):
    seed_info: Dict[str, Any]
    breaked: bool

    # Stage 1 outputs
    behavior_tree: Optional[str]
    npc_tools: Optional[str]
    npc_task: Optional[str]
    npc_policy: Optional[str]

    # Stage 2 outputs
    policy_str: Optional[str]
    test_cases: Optional[List[str]]

    # Stage 3 outputs
    checked_tools: Optional[List[Dict[str, Any]]]
    tasks_and_backgrounds: Optional[List[Dict[str, Any]]]


_STEP_CONFIG_KEYS = ("model_name", "temperature", "max_tokens")


def create_step_config(base_config: RunnableConfig, step_name: str) -> RunnableConfig:
    step_model = base_config["configurable"]["step_models"][step_name]
    configurable = {k: v for k, v in step_model.items() if k in _STEP_CONFIG_KEYS}
    configurable["api_configs"] = base_config["configurable"]["api_configs"]
    return {"configurable": configurable}


def npc_behavior_tree_node(state: NPCDataGenState, config: RunnableConfig):
    """Stage 1: 根据NPC角色种子生成行为树和工具集"""
    from functions.npc_tool_set_policy_gen import generate_npc_behavior_tree

    step_config = create_step_config(config, "BehaviorTreeAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)

    npc_persona = state["seed_info"]["persona"]
    normal_workflow = state["seed_info"].get("normal_workflow")
    game_context = state["seed_info"].get("game_context", {})
    npc_type = state["seed_info"].get("npc_type", "generic")

    all_content, task, tools, policy = generate_npc_behavior_tree(
        cfg=cfg,
        npc_persona=npc_persona,
        normal_workflow=normal_workflow,
        game_context=game_context,
        npc_type=npc_type,
    )

    if task is None or tools is None or policy is None:
        return {
            "breaked": True,
            "behavior_tree": None,
            "npc_task": None,
            "npc_tools": None,
            "npc_policy": None,
        }

    return {
        "behavior_tree": all_content,
        "npc_task": task,
        "npc_tools": tools,
        "npc_policy": policy,
    }


def npc_test_case_node(state: NPCDataGenState, config: RunnableConfig):
    """Stage 2: 从行为树生成测试用例（含正常路径和exploit路径）"""
    if state["breaked"]:
        return {"policy_str": None, "test_cases": None, "breaked": True}

    from functions.npc_policy_task import generate_npc_test_cases

    step_config = create_step_config(config, "TestCaseAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)

    task_description = state["npc_task"]
    policy_tree = state["npc_policy"]
    tools = state["npc_tools"]

    try:
        tools_parsed = json.loads(tools)
    except (json.JSONDecodeError, TypeError):
        print(f"Error: tools is not valid JSON: {tools}")
        return {"breaked": True, "policy_str": None, "test_cases": None}

    all_content, policy, test_cases = generate_npc_test_cases(
        cfg=cfg,
        task_description=task_description,
        policy_tree=policy_tree,
        tools=tools_parsed,
        npc_type=state["seed_info"].get("npc_type", "generic"),
    )

    return {
        "checked_tools": tools_parsed,
        "policy_str": policy,
        "test_cases": test_cases,
        "breaked": not test_cases or policy is None,
    }


def npc_refine_node(state: NPCDataGenState, config: RunnableConfig):
    """Stage 3: 精炼测试用例为自然对话场景（可选，失败则使用Stage 2原始数据）"""
    if state["breaked"]:
        return {"tasks_and_backgrounds": None}

    test_cases = state["test_cases"]
    if not test_cases:
        return {"breaked": True, "tasks_and_backgrounds": None}

    from functions.npc_refine_task import refine_npc_test_cases

    step_config = create_step_config(config, "RefineAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)

    tools = state["checked_tools"]

    try:
        refined_tasks = refine_npc_test_cases(cfg, tools, test_cases)
    except Exception as e:
        print(f"  [Stage3] Refine failed: {e}, using raw test cases")
        refined_tasks = []

    # 如果精炼失败，回退使用Stage 2的原始test_cases
    if not refined_tasks:
        print(f"  [Stage3] Falling back to raw test cases ({len(test_cases)} items)")
        refined_tasks = test_cases

    return {"tasks_and_backgrounds": refined_tasks}


# Build the graph
builder = StateGraph(NPCDataGenState, config_schema=RunnableConfig)
builder.add_node("behavior_tree_gen", npc_behavior_tree_node)
builder.add_node("test_case_gen", npc_test_case_node)
builder.add_node("refine", npc_refine_node)

builder.set_entry_point("behavior_tree_gen")
builder.add_edge("behavior_tree_gen", "test_case_gen")
builder.add_edge("test_case_gen", "refine")
graph = builder.compile()


def run_npc_data_gen(seed_info: dict, run_config: dict):
    """运行NPC数据生成pipeline"""
    output_path = run_config["logging"]["task_file_path"]

    # 如果是相对路径，基于npc_dialogue/目录解析
    if not os.path.isabs(output_path):
        npc_dialogue_root = os.path.join(os.path.dirname(__file__), '..')
        output_path = os.path.normpath(os.path.join(npc_dialogue_root, output_path))

    log_dir = os.path.dirname(output_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    runnable_config = {"configurable": run_config or {}}
    runnable_config["recursion_limit"] = 100

    initial_state = {
        "seed_info": seed_info,
        "breaked": False,
    }

    final_state = graph.invoke(initial_state, config=runnable_config)

    if not final_state["breaked"] and final_state["tasks_and_backgrounds"]:
        save_data = {
            "id": seed_info["id"],
            "npc_type": seed_info.get("npc_type"),
            "checked_tools": final_state["checked_tools"],
            "policy": final_state["policy_str"],
            "tasks_and_backgrounds": final_state["tasks_and_backgrounds"],
        }

        with log_file_lock:
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(save_data, ensure_ascii=False) + '\n')

    return final_state


if __name__ == "__main__":
    with open("../configs/data_gen.yaml", 'r', encoding='utf-8') as f:
        agent_config = yaml.safe_load(f)

    with open("../configs/npc_personas.jsonl", 'r', encoding='utf-8') as f:
        personas = [json.loads(line) for line in f]

    for persona in personas[:5]:  # Test with first 5
        seed = {
            "id": persona["id"],
            "persona": persona["persona"],
            "npc_type": persona["npc_type"],
            "normal_workflow": persona.get("normal_workflow"),
            "game_context": persona.get("game_context", {}),
        }
        run_npc_data_gen(seed, run_config=agent_config)
