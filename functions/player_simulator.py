"""
对抗性玩家模拟函数
模拟玩家回复（支持正常玩家和exploit玩家两种模式）
"""

import json
from configuration import ModelConfiguration
from prompts.player_simulation_prompt import (
    PLAYER_SIMULATION_PROMPT,
    PLAYER_NORMAL_PROMPT,
)


def mock_player_response(
    cfg: ModelConfiguration,
    player_background: str,
    actual_player_state: str,
    player_goal: str,
    exploit_strategy: str,
    conversation_context: str,
    is_adversarial: bool = True,
) -> str:
    """
    模拟玩家的回复。

    Args:
        cfg: 模型配置
        player_background: 玩家角色背景
        actual_player_state: 玩家真实游戏状态
        player_goal: 玩家目标
        exploit_strategy: 玩家exploit策略（adversarial模式下使用）
        conversation_context: 当前对话上下文
        is_adversarial: 是否为对抗性模式

    Returns:
        玩家的回复文本。如果包含 ###STOP 表示对话结束。
    """
    from openai import OpenAI

    if is_adversarial:
        prompt = PLAYER_SIMULATION_PROMPT.format(
            player_background=player_background,
            actual_player_state=actual_player_state,
            player_goal=player_goal,
            exploit_strategy=exploit_strategy,
            conversation_context=conversation_context,
        )
    else:
        prompt = PLAYER_NORMAL_PROMPT.format(
            player_background=player_background,
            actual_player_state=actual_player_state,
            player_goal=player_goal,
            conversation_context=conversation_context,
        )

    client = OpenAI(base_url=cfg.api_base, api_key=cfg.api_key)

    try:
        response = client.chat.completions.create(
            model=cfg.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "算了，不聊了。###STOP"
