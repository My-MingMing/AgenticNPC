"""
游戏工具模拟函数
使用LLM模拟游戏工具的调用返回，保证世界状态一致性
"""

import json
from typing import Optional
from configuration import ModelConfiguration
from prompts.tool_simulation_prompt import GAME_TOOL_SIMULATION_PROMPT


def mock_tool_response(
    cfg: ModelConfiguration,
    tool_name: str,
    tool_input: dict,
    world_state: dict,
    tool_definitions: str,
    interaction_history: str,
) -> str:
    """
    模拟游戏工具的响应。

    Args:
        cfg: 模型配置
        tool_name: 调用的工具名
        tool_input: 工具输入参数
        world_state: 预定义的游戏世界状态（tool_return_expected）
        tool_definitions: 工具定义JSON字符串
        interaction_history: 之前的交互历史

    Returns:
        工具调用的JSON结果字符串
    """
    from openai import OpenAI
    import re

    prompt = GAME_TOOL_SIMULATION_PROMPT.format(
        world_state=json.dumps(world_state, ensure_ascii=False, indent=2),
        tool_definitions=tool_definitions,
        tool_name=tool_name,
        tool_input=json.dumps(tool_input, ensure_ascii=False),
        interaction_history=interaction_history or "无",
    )

    client = OpenAI(base_url=cfg.api_base, api_key=cfg.api_key)

    try:
        response = client.chat.completions.create(
            model=cfg.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )
        content = response.choices[0].message.content
    except Exception as e:
        return json.dumps({"error": f"Tool simulation failed: {str(e)}"})

    # 提取 <generated_tool_response> 标签内的内容
    match = re.search(
        r'<generated_tool_response>(.*?)(?:</generated_tool_response>|$)',
        content, re.DOTALL
    )
    if match:
        result = match.group(1).strip()
    else:
        result = content.strip()

    # 尝试验证是否为合法JSON
    try:
        json.loads(result)
    except json.JSONDecodeError:
        # 如果不是合法JSON，包装一下
        result = json.dumps({"response": result})

    return result
