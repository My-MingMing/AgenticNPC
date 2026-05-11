"""
游戏Mock工具（RL训练环境）
在RL rollout过程中模拟游戏工具和玩家交互
"""

import json
import asyncio
from typing import Dict, Any, Optional

from prompts.tool_simulation_prompt import GAME_TOOL_SIMULATION_PROMPT_FOR_RL
from prompts.player_simulation_prompt import PLAYER_SIMULATION_PROMPT_FOR_RL


class GameMockTool:
    """
    游戏环境模拟工具，用于RL训练中的Agent交互。

    功能：
    1. 模拟游戏工具调用（query_player_info, execute_transaction等）
    2. 模拟对抗性玩家行为
    3. 维护世界状态一致性
    """

    def __init__(self, llm_client, model_name: str, max_concurrent: int = 16):
        self.llm_client = llm_client
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def execute(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        world_state: Dict[str, Any],
        interaction_history: str,
        tool_definitions: str,
    ) -> str:
        """
        执行工具模拟调用。

        Args:
            tool_name: 工具名称（如 "query_player_info" 或 "mock_user"）
            tool_input: 工具输入参数
            world_state: 游戏世界状态（预定义）
            interaction_history: 之前的交互历史
            tool_definitions: 可用工具定义

        Returns:
            模拟的工具响应字符串
        """
        async with self.semaphore:
            if tool_name == "mock_user":
                return await self._simulate_player(tool_input, world_state, interaction_history)
            else:
                return await self._simulate_game_tool(
                    tool_name, tool_input, world_state, interaction_history, tool_definitions
                )

    async def _simulate_game_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        world_state: Dict[str, Any],
        interaction_history: str,
        tool_definitions: str,
    ) -> str:
        """模拟游戏系统工具的响应"""
        prompt = GAME_TOOL_SIMULATION_PROMPT_FOR_RL.format(
            world_state=json.dumps(world_state, ensure_ascii=False, indent=2),
            tool_name=tool_name,
            tool_input=json.dumps(tool_input, ensure_ascii=False),
            interaction_history=interaction_history,
        )

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024,
            )
            content = response.choices[0].message.content

            # 提取工具响应
            import re
            match = re.search(r'<generated_tool_response>(.*?)(?:</generated_tool_response>|$)', content, re.DOTALL)
            if match:
                return match.group(1).strip()
            return content.strip()

        except Exception as e:
            return json.dumps({"error": f"Tool simulation failed: {str(e)}"})

    async def _simulate_player(
        self,
        tool_input: Dict[str, Any],
        world_state: Dict[str, Any],
        interaction_history: str,
    ) -> str:
        """模拟对抗性玩家的回复"""
        player_bg = world_state.get("player_background", "")
        actual_state = world_state.get("actual_player_state", "")
        player_goal = world_state.get("player_goal", "正常完成交互")
        exploit_strategy = world_state.get("exploit_strategy", "")

        prompt = PLAYER_SIMULATION_PROMPT_FOR_RL.format(
            player_background=player_bg,
            actual_player_state=actual_state,
            player_goal=player_goal,
            exploit_strategy=exploit_strategy,
            conversation_context=interaction_history,
        )

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return "（玩家沉默了一会）算了，不买了。###STOP"


class GameWorldStateTracker:
    """
    游戏世界状态追踪器。
    在RL训练中追踪工具调用导致的状态变化。
    """

    def __init__(self, initial_state: Dict[str, Any]):
        self.state = initial_state.copy()
        self.deltas = []

    def apply_delta(self, tool_name: str, tool_input: Dict[str, Any], tool_output: Dict[str, Any]):
        """根据工具调用结果更新世界状态"""
        if tool_name == "execute_transaction" and tool_output.get("success"):
            # 更新金币余额
            if "new_gold_balance" in tool_output:
                self.state["player_gold"] = tool_output["new_gold_balance"]
            self.deltas.append({
                "tool": tool_name,
                "change": f"gold -> {tool_output.get('new_gold_balance')}",
            })

        elif tool_name == "assign_quest" and tool_output.get("success"):
            quest_id = tool_input.get("quest_id")
            if quest_id:
                active_quests = self.state.get("active_quests", [])
                active_quests.append(quest_id)
                self.state["active_quests"] = active_quests
            self.deltas.append({
                "tool": tool_name,
                "change": f"quest {quest_id} assigned",
            })

        elif tool_name == "modify_relationship" and tool_output.get("new_relationship_value") is not None:
            target = tool_input.get("target_id")
            self.state.setdefault("relationships", {})[target] = tool_output["new_relationship_value"]
            self.deltas.append({
                "tool": tool_name,
                "change": f"{target} -> {tool_output['new_relationship_value']}",
            })

    def get_current_state(self) -> Dict[str, Any]:
        return self.state.copy()

    def get_deltas(self) -> list:
        return self.deltas.copy()
