"""
NPC任务求解函数
负责调用LLM进行推理，解析输出中的tool_call/question/answer标签
"""

import re
import json
from typing import Optional, Tuple
from configuration import ModelConfiguration


def solve_task_by_tools(cfg: ModelConfiguration, messages: list) -> Tuple[str, str, Optional[dict]]:
    """
    调用LLM进行一步推理。解析输出中的标签决定下一步动作。

    Args:
        cfg: 模型配置
        messages: 完整的对话历史（含system prompt）

    Returns:
        (raw_response, action_type, action_data)
        action_type: "tool_call" | "question" | "answer" | "transfer" | "error"
        action_data: 解析出的结构化数据（tool调用参数、问题内容等）
    """
    from openai import OpenAI

    client = OpenAI(base_url=cfg.api_base, api_key=cfg.api_key)

    try:
        response = client.chat.completions.create(
            model=cfg.model_name,
            messages=messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        content = response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return str(e), "error", None

    action_type, action_data = _parse_agent_output(content)
    return content, action_type, action_data


def _parse_agent_output(content: str) -> Tuple[str, Optional[dict]]:
    """
    解析Agent输出，提取动作类型和数据。

    支持的标签:
    - <tool_call>{"name": "xxx", "arguments": {...}}</tool_call> → 工具调用
    - <question>问题内容</question> → 向玩家提问
    - <answer>最终回复</answer> → 对话结束
    - ###TRANSFER_TO_HUMAN → 转人工
    """
    # 检查终止信号
    if "###TRANSFER_TO_HUMAN" in content:
        return "transfer", {"reason": "escalated to human"}

    # 检查tool_call
    tool_match = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
    if tool_match:
        try:
            tool_data = json.loads(tool_match.group(1).strip())
            return "tool_call", tool_data
        except json.JSONDecodeError:
            # 尝试修复常见的JSON格式问题
            raw = tool_match.group(1).strip()
            return "tool_call", {"raw": raw, "parse_error": True}

    # 检查question（向玩家提问）
    question_match = re.search(r'<question>(.*?)</question>', content, re.DOTALL)
    if question_match:
        return "question", {"content": question_match.group(1).strip()}

    # 检查answer（对话结束）
    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    if answer_match:
        return "answer", {"content": answer_match.group(1).strip()}

    # 检查###STOP
    if "###STOP" in content:
        return "answer", {"content": content.replace("###STOP", "").strip()}

    # 无法识别的格式，当作普通回复处理
    return "answer", {"content": content.strip()}
