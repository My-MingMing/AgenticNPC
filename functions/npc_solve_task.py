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
        request_kwargs = {
            "model": cfg.model_name,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        # Qwen-compatible gateways differ in how they expose thinking. Only
        # send the flag when explicitly enabled so older OpenAI-compatible
        # endpoints remain usable.
        extra_body = {"enable_thinking": bool(cfg.enable_thinking)}
        if cfg.enable_thinking:
            extra_body["thinking_budget"] = min(max(cfg.thinking_budget, 1), 512)
        request_kwargs["extra_body"] = extra_body
        response = client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return str(e), "error", None

    action_type, action_data = _parse_agent_output(
        content, final_answer_max_tokens=cfg.final_answer_max_tokens
    )
    if action_type == "answer" and isinstance(action_data, dict):
        # Never expose hidden reasoning or an overlong answer to the player.
        # Tool-call responses remain untouched because their raw JSON is
        # consumed by the environment, not shown as NPC prose.
        answer_text = action_data.get("content", "")
        raw_response = f"<answer>{answer_text}</answer>\n###STOP"
    if action_type == "answer" and isinstance(action_data, dict):
        return raw_response, action_type, action_data
    return content, action_type, action_data


def _truncate_visible_answer(text: str, max_tokens: int) -> str:
    """Conservative visible-answer cap without touching tool-call tokens.

    Exact tokenizer access is intentionally avoided in the online parser. A
    UTF-8/ASCII hybrid estimate is sufficient for the product boundary; the
    training tokenizer remains authoritative for RL length accounting.
    """
    limit = max(1, int(max_tokens))
    # Chinese characters are close to one token, while English words average
    # several characters per token. Keep a small safety margin.
    # One Unicode code point is a conservative display proxy for one token in
    # Chinese-heavy NPC dialogue. Exact token counts are still measured by the
    # Qwen tokenizer in rollout/evaluation.
    budget_chars = limit
    if len(text) <= budget_chars:
        return text.strip()
    return text[:budget_chars].rstrip() + "…"


def _parse_agent_output(content: str, final_answer_max_tokens: int = 128) -> Tuple[str, Optional[dict]]:
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
        return "answer", {"content": _truncate_visible_answer(answer_match.group(1), final_answer_max_tokens)}

    # 检查###STOP
    if "###STOP" in content:
        return "answer", {"content": _truncate_visible_answer(content.replace("###STOP", ""), final_answer_max_tokens)}

    # 无法识别的格式，当作普通回复处理
    return "answer", {"content": _truncate_visible_answer(content, final_answer_max_tokens)}
