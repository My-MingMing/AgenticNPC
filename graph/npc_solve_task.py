"""
NPC任务求解 - LangGraph工作流
实现 Agent求解循环: reason_and_act → mock_tools/mock_user → 循环直到结束
"""

import os
import json
import sys
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configuration import ModelConfiguration


class SolveTaskState(TypedDict):
    task_info: Dict[str, Any]
    messages: List[Dict[str, str]]
    solve_history: List[str]
    tool_call_history: List[Dict[str, Any]]
    turn_count: int
    max_turns: int
    terminated: bool
    final_trajectory: Optional[str]


_STEP_CONFIG_KEYS = ("model_name", "temperature", "max_tokens")


def create_step_config(base_config: RunnableConfig, step_name: str) -> RunnableConfig:
    step_model = base_config["configurable"]["step_models"][step_name]
    configurable = {k: v for k, v in step_model.items() if k in _STEP_CONFIG_KEYS}
    configurable["api_configs"] = base_config["configurable"]["api_configs"]
    return {"configurable": configurable}


def reason_and_act_node(state: SolveTaskState, config: RunnableConfig):
    """Agent推理节点：调用LLM生成下一步动作"""
    from functions.npc_solve_task import solve_task_by_tools

    step_config = create_step_config(config, "SolveAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)

    raw_response, action_type, action_data = solve_task_by_tools(cfg, state["messages"])

    # 将Agent回复加入历史
    messages = state["messages"] + [{"role": "assistant", "content": raw_response}]
    solve_history = state["solve_history"] + [raw_response]
    turn_count = state["turn_count"] + 1

    # 检查终止条件
    terminated = (
        action_type in ("answer", "transfer", "error")
        or turn_count >= state["max_turns"]
    )

    return {
        "messages": messages,
        "solve_history": solve_history,
        "turn_count": turn_count,
        "terminated": terminated,
        "_action_type": action_type,
        "_action_data": action_data,
    }


def mock_tools_node(state: SolveTaskState, config: RunnableConfig):
    """工具模拟节点：模拟游戏工具的响应"""
    from functions.game_mock_tools import mock_tool_response

    step_config = create_step_config(config, "MockToolAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)

    action_data = state.get("_action_data", {})
    task_info = state["task_info"]

    tool_name = action_data.get("name", "unknown")
    tool_args = action_data.get("arguments", action_data.get("parameters", {}))

    # 从task_info获取世界状态
    world_state = task_info.get("tool_return_expected", {})
    tool_definitions = json.dumps(task_info.get("tools", []), ensure_ascii=False)

    # 构建交互历史摘要
    history_summary = "\n".join(state["solve_history"][-4:])

    tool_result = mock_tool_response(
        cfg=cfg,
        tool_name=tool_name,
        tool_input=tool_args,
        world_state=world_state,
        tool_definitions=tool_definitions,
        interaction_history=history_summary,
    )

    # 将工具结果加入消息
    tool_msg = f"<tool_response>\n{tool_result}\n</tool_response>"
    messages = state["messages"] + [{"role": "user", "content": tool_msg}]
    tool_call_history = state["tool_call_history"] + [{
        "tool_name": tool_name,
        "input": tool_args,
        "output": tool_result,
    }]

    return {
        "messages": messages,
        "tool_call_history": tool_call_history,
    }


def mock_user_node(state: SolveTaskState, config: RunnableConfig):
    """玩家模拟节点：模拟玩家对NPC提问的回复"""
    from functions.player_simulator import mock_player_response

    step_config = create_step_config(config, "MockUserAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)

    task_info = state["task_info"]
    action_data = state.get("_action_data", {})

    # 构建对话上下文
    conversation = "\n".join([
        f"{'NPC' if m['role']=='assistant' else '玩家'}: {m['content'][:200]}"
        for m in state["messages"][-6:]
    ])

    is_adversarial = task_info.get("test_type") == "adversarial"

    player_reply = mock_player_response(
        cfg=cfg,
        player_background=task_info.get("user_background", ""),
        actual_player_state=json.dumps(task_info.get("actual_player_state", {}), ensure_ascii=False),
        player_goal=task_info.get("player_goal", "正常完成交互"),
        exploit_strategy=task_info.get("user_escape_strategy", ""),
        conversation_context=conversation,
        is_adversarial=is_adversarial,
    )

    # 检查玩家是否结束对话
    if "###STOP" in player_reply:
        player_reply = player_reply.replace("###STOP", "").strip()
        messages = state["messages"] + [{"role": "user", "content": player_reply}]
        return {"messages": messages, "terminated": True}

    messages = state["messages"] + [{"role": "user", "content": player_reply}]
    return {"messages": messages}


def route_action(state: SolveTaskState) -> str:
    """根据Agent的动作类型路由到对应节点"""
    if state.get("terminated"):
        return "end"
    action_type = state.get("_action_type", "answer")
    if action_type == "tool_call":
        return "mock_tools"
    elif action_type == "question":
        return "mock_user"
    else:
        return "end"


# Build the graph
builder = StateGraph(SolveTaskState, config_schema=RunnableConfig)
builder.add_node("reason_and_act", reason_and_act_node)
builder.add_node("mock_tools", mock_tools_node)
builder.add_node("mock_user", mock_user_node)

builder.set_entry_point("reason_and_act")
builder.add_conditional_edges("reason_and_act", route_action, {
    "mock_tools": "mock_tools",
    "mock_user": "mock_user",
    "end": END,
})
builder.add_edge("mock_tools", "reason_and_act")
builder.add_edge("mock_user", "reason_and_act")

solve_graph = builder.compile()


def run_solve_task(task_info: dict, npc_policy: str, tools: list, config: dict, max_turns: int = 10) -> dict:
    """
    对单个任务运行Agent求解。

    Args:
        task_info: 任务信息（含task, user_background, tool_return_expected等）
        npc_policy: NPC策略文档（System Prompt）
        tools: 工具定义列表
        config: 运行配置
        max_turns: 最大对话轮数

    Returns:
        求解结果dict，含messages, tool_call_history, terminated
    """
    # 构建初始消息
    system_msg = npc_policy
    user_msg = task_info.get("task", "")

    initial_messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # 将工具和世界状态注入task_info
    task_info_full = {**task_info, "tools": tools}

    initial_state = {
        "task_info": task_info_full,
        "messages": initial_messages,
        "solve_history": [],
        "tool_call_history": [],
        "turn_count": 0,
        "max_turns": max_turns,
        "terminated": False,
        "final_trajectory": None,
    }

    runnable_config = {"configurable": config, "recursion_limit": 50}
    final_state = solve_graph.invoke(initial_state, config=runnable_config)

    return {
        "messages": final_state["messages"],
        "tool_call_history": final_state["tool_call_history"],
        "solve_history": final_state["solve_history"],
        "turn_count": final_state["turn_count"],
        "terminated": final_state["terminated"],
    }
