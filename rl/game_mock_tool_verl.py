"""
verl兼容的游戏Mock工具
遵循verl BaseTool接口：create, execute, calc_reward, release
"""

import json
import asyncio
import os
from typing import Dict, Any, Optional

from prompts.tool_simulation_prompt import GAME_TOOL_SIMULATION_PROMPT_FOR_RL
from prompts.player_simulation_prompt import PLAYER_SIMULATION_PROMPT_FOR_RL


class GameMockSandboxTool:
    """
    verl兼容的游戏环境模拟工具。

    实现verl BaseTool接口:
    - create(create_kwargs) → 初始化per-instance状态
    - execute(tool_name, tool_input, ...) → 模拟工具/玩家响应
    - calc_reward(solution, ...) → 计算奖励分数
    - release() → 清理资源
    """

    def __init__(self, num_workers=8, rate_limit=8, acquire_timeout=90, default_timeout=90, **kwargs):
        self.num_workers = num_workers
        self.rate_limit = rate_limit
        self.acquire_timeout = acquire_timeout
        self.default_timeout = default_timeout
        self.semaphore = asyncio.Semaphore(rate_limit)

        # LLM client配置
        self.api_base = os.getenv("MOCK_TOOL_API_BASE", "")
        self.api_key = os.getenv("MOCK_TOOL_API_KEY", "")
        self.model_name = os.getenv("MOCK_TOOL_MODEL", "qwen-plus")

    def create(self, create_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """初始化per-instance状态（每个RL trajectory一个实例）"""
        return {
            "tool_return_expected": create_kwargs.get("tool_return_expected", {}),
            "user_background": create_kwargs.get("user_background", ""),
            "exploit_strategy": create_kwargs.get("exploit_strategy", ""),
            "player_goal": create_kwargs.get("player_goal", ""),
            "actual_player_state": create_kwargs.get("actual_player_state", {}),
            "test_type": create_kwargs.get("test_type", "normal"),
            "interaction_history": [],
        }

    async def execute(self, tool_name: str, tool_input: str, instance_state: Dict[str, Any],
                      history: str = "", **kwargs) -> str:
        """执行工具调用或玩家模拟"""
        async with self.semaphore:
            if tool_name == "mock_user":
                return await self._mock_player(tool_input, instance_state, history)
            else:
                return await self._mock_game_tool(tool_name, tool_input, instance_state, history)

    async def _mock_game_tool(self, tool_name: str, tool_input: str,
                              state: Dict[str, Any], history: str) -> str:
        """模拟游戏系统工具响应"""
        from openai import AsyncOpenAI

        try:
            input_data = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
        except json.JSONDecodeError:
            input_data = {"raw": tool_input}

        prompt = GAME_TOOL_SIMULATION_PROMPT_FOR_RL.format(
            world_state=json.dumps(state["tool_return_expected"], ensure_ascii=False, indent=2),
            tool_name=tool_name,
            tool_input=json.dumps(input_data, ensure_ascii=False),
            interaction_history=history[-2000:] if history else "None",
        )

        client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=1024,
                ),
                timeout=self.default_timeout,
            )
            content = resp.choices[0].message.content
            import re
            match = re.search(r'<generated_tool_response>(.*?)(?:</generated_tool_response>|$)', content, re.DOTALL)
            return match.group(1).strip() if match else content.strip()
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _mock_player(self, tool_input: str, state: Dict[str, Any], history: str) -> str:
        """模拟玩家响应"""
        from openai import AsyncOpenAI

        is_adversarial = state.get("test_type") in ("adversarial", "exploit")

        prompt = PLAYER_SIMULATION_PROMPT_FOR_RL.format(
            player_background=state.get("user_background", ""),
            actual_player_state=json.dumps(state.get("actual_player_state", {}), ensure_ascii=False),
            player_goal=state.get("player_goal", "complete interaction"),
            exploit_strategy=state.get("exploit_strategy", "") if is_adversarial else "None - normal player",
            conversation_context=history[-2000:] if history else "No prior conversation",
        )

        client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=512,
                ),
                timeout=self.default_timeout,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return "算了，不聊了。###STOP"

    async def calc_reward(self, solution_str: str, ground_truth: str,
                          extra_info: Dict[str, Any], **kwargs) -> float:
        """计算奖励分数"""
        from rl.npc_reward_function import compute_score_npc_dialogue
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)

        score = await compute_score_npc_dialogue(
            data_source="npc_dialogue",
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            llm_client=client,
        )
        return score

    def release(self):
        """清理资源"""
        pass
