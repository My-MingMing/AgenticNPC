"""
verl兼容的游戏Mock工具
遵循verl BaseTool接口：create, execute, calc_reward, release
"""

import json
import asyncio
import os
import copy
from typing import Dict, Any, Optional

from prompts.tool_simulation_prompt import GAME_TOOL_SIMULATION_PROMPT_FOR_RL
from prompts.player_simulation_prompt import PLAYER_SIMULATION_PROMPT_FOR_RL
from rl.tool_policy import normalize_tool_call


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
        expected = copy.deepcopy(create_kwargs.get("tool_return_expected", {}))
        return {
            "tool_return_expected": expected,
            "user_background": create_kwargs.get("user_background", ""),
            "exploit_strategy": create_kwargs.get("exploit_strategy", ""),
            "player_goal": create_kwargs.get("player_goal", ""),
            "actual_player_state": create_kwargs.get("actual_player_state", {}),
            "test_type": create_kwargs.get("test_type", "normal"),
            "interaction_history": [],
            "event_log": [],
            "runtime_state": copy.deepcopy(create_kwargs.get("actual_player_state", {})),
        }

    async def execute(self, tool_name: str, tool_input: str, instance_state: Dict[str, Any],
                      history: str = "", **kwargs) -> str:
        """执行工具调用或玩家模拟"""
        async with self.semaphore:
            decoded_input = self._decode_tool_input(tool_input)
            call = (
                normalize_tool_call(
                    decoded_input
                    if isinstance(decoded_input, dict) and ("name" in decoded_input or "tool_name" in decoded_input)
                    else {"name": "mock_tool", "arguments": decoded_input}
                )
                if tool_name == "mock_tool"
                else normalize_tool_call({"name": tool_name, "arguments": decoded_input})
            )
            if call.name == "mock_user":
                result = await self._mock_player(json.dumps(call.arguments), instance_state, history)
            elif not call.valid:
                result = json.dumps({"error": "invalid_tool_call"}, ensure_ascii=False)
            else:
                result = await self._mock_game_tool(call.name, call.arguments, instance_state, history)
            instance_state.setdefault("interaction_history", []).append({
                "tool_name": call.name,
                "arguments": call.arguments,
                "result": result,
            })
            return result

    @staticmethod
    def _decode_tool_input(tool_input: Any) -> Dict[str, Any]:
        if isinstance(tool_input, dict):
            # veRL exposes one outer mock_tool(query=<inner JSON>) function.
            if "query" in tool_input:
                inner = tool_input["query"]
                try:
                    return json.loads(inner) if isinstance(inner, str) else inner
                except json.JSONDecodeError:
                    return {"raw": inner}
            return tool_input
        if isinstance(tool_input, str):
            try:
                value = json.loads(tool_input)
                return value if isinstance(value, dict) else {"raw": value}
            except json.JSONDecodeError:
                return {"raw": tool_input}
        return {"raw": tool_input}

    @staticmethod
    def _matches(expected_input: Any, actual_input: Dict[str, Any]) -> bool:
        if not isinstance(expected_input, dict):
            return True
        return all(actual_input.get(key) == value for key, value in expected_input.items())

    def _resolve_expected(self, tool_name: str, tool_input: Dict[str, Any], state: Dict[str, Any]):
        expected = state.get("tool_return_expected", {})
        # Warm-up data uses a direct tool_name -> result mapping.
        if tool_name in expected and not isinstance(expected.get(tool_name), list):
            result = copy.deepcopy(expected[tool_name])
            return self._overlay_runtime_state(tool_name, result, state)
        # Agentic data stores ordered normal/hack paths. Only the normal path is an oracle.
        for step in expected.get("normal_path", []):
            if step.get("tool_name") == tool_name and self._matches(step.get("input"), tool_input):
                result = copy.deepcopy(step.get("expected_output", {}))
                return self._overlay_runtime_state(tool_name, result, state)
        return None

    @staticmethod
    def _overlay_runtime_state(tool_name: str, result: Any, state: Dict[str, Any]):
        if not isinstance(result, dict):
            return result
        runtime = state.get("runtime_state", {})
        if tool_name == "query_player_info" and "gold_balance" in runtime:
            result["gold_balance"] = runtime["gold_balance"]
            if isinstance(result.get("balance"), dict):
                result["balance"]["gold_balance"] = runtime["gold_balance"]
        if tool_name == "query_quest_status" and runtime.get("active_quests"):
            result["active_quests"] = list(runtime["active_quests"])
        if tool_name == "query_player_info" and runtime.get("relationships"):
            result["relationships"] = copy.deepcopy(runtime["relationships"])
        return result

    @staticmethod
    def _apply_state_delta(
        tool_name: str,
        tool_input: Dict[str, Any],
        output: Any,
        state: Dict[str, Any],
        source: str,
    ):
        if not isinstance(output, dict):
            return
        runtime = state.setdefault("runtime_state", {})
        if tool_name == "execute_transaction" and output.get("success"):
            if "new_gold_balance" in output:
                runtime["gold_balance"] = output["new_gold_balance"]
        elif tool_name == "assign_quest" and output.get("success"):
            runtime.setdefault("active_quests", []).append(tool_input.get("quest_id"))
        elif tool_name == "modify_relationship":
            new_value = output.get("new_relationship_value", output.get("new_value"))
            if new_value is not None:
                runtime.setdefault("relationships", {})[
                    tool_input.get("target_id", "unknown")
                ] = new_value
        state.setdefault("event_log", []).append({
            "tool_name": tool_name,
            "arguments": tool_input,
            "output": copy.deepcopy(output),
            "source": source,
        })

    async def _mock_game_tool(self, tool_name: str, tool_input: Dict[str, Any],
                              state: Dict[str, Any], history: str) -> str:
        """模拟游戏系统工具响应"""
        from openai import AsyncOpenAI

        deterministic = self._resolve_expected(tool_name, tool_input, state)
        if deterministic is not None:
            self._apply_state_delta(
                tool_name, tool_input, deterministic, state, source="scenario_oracle"
            )
            # Keep provenance internal. The actor sees the same tool schema in
            # training and serving instead of a training-only {source, data} wrapper.
            return json.dumps(deterministic, ensure_ascii=False)

        prompt = GAME_TOOL_SIMULATION_PROMPT_FOR_RL.format(
            world_state=json.dumps(state["tool_return_expected"], ensure_ascii=False, indent=2),
            tool_name=tool_name,
            tool_input=json.dumps(tool_input, ensure_ascii=False),
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
            result = match.group(1).strip() if match else content.strip()
            try:
                parsed_result = json.loads(result)
            except json.JSONDecodeError:
                parsed_result = {"message": result}
            self._apply_state_delta(
                tool_name, tool_input, parsed_result, state, source="llm_fallback"
            )
            return json.dumps(parsed_result, ensure_ascii=False)
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

        judge_base = os.getenv("JUDGE_API_BASE", "")
        judge_key = os.getenv("JUDGE_API_KEY", "")
        use_judge = os.getenv("NPC_USE_LLM_JUDGE", "false").lower() == "true"
        client = AsyncOpenAI(base_url=judge_base, api_key=judge_key) if use_judge and judge_base else None

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
