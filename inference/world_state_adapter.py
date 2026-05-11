"""
游戏世界状态适配器
将游戏后端数据库状态转换为AgenticQwen框架的 tool_return_expected 格式
用于训练时的Mock环境构建和推理时的真实API对接
"""

import json
from typing import Dict, Any, Optional


class WorldStateAdapter:
    """
    将游戏DB状态转换为NPC对话所需的世界状态格式。

    训练模式: 从预定义JSON加载模拟状态
    生产模式: 从游戏数据库实时查询
    """

    def __init__(self, mode: str = "mock"):
        """
        Args:
            mode: "mock" (训练/测试) 或 "production" (对接真实游戏DB)
        """
        self.mode = mode
        self._game_db_client = None

    def set_production_client(self, db_client):
        """设置生产环境的数据库客户端"""
        self._game_db_client = db_client
        self.mode = "production"

    def get_world_state(
        self,
        player_id: str,
        npc_id: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        获取当前世界状态。

        Returns:
            包含以下字段的字典:
            - player_info: 玩家基本信息
            - npc_inventory: NPC库存
            - quest_data: 任务数据
            - relationship_data: 关系数据
            - world_context: 世界上下文（时间、天气等）
        """
        if self.mode == "mock":
            return self._get_mock_state(player_id, npc_id, extra_context)
        else:
            return self._get_production_state(player_id, npc_id)

    def _get_mock_state(
        self,
        player_id: str,
        npc_id: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """从预定义数据构造模拟世界状态（训练用）"""
        if extra_context:
            return extra_context

        # 默认模拟状态
        return {
            "player_info": {
                "player_id": player_id,
                "level": 12,
                "class": "warrior",
                "gold": 500,
                "faction_reputation": {
                    "mountain_dwarves": "friendly",
                    "forest_elves": "neutral",
                    "underground_network": "unfriendly",
                },
                "completed_quests": ["tutorial_001", "forest_exploration"],
                "active_quests": ["gather_ore_003"],
            },
            "npc_inventory": {
                "npc_id": npc_id,
                "items": [
                    {"id": "iron_sword_01", "name": "铁剑", "price": 150, "level_req": 5, "quantity": 3},
                    {"id": "steel_shield_02", "name": "钢盾", "price": 300, "level_req": 10, "quantity": 2},
                    {"id": "mithril_axe_03", "name": "秘银战斧", "price": 800, "level_req": 20, "quantity": 1},
                ],
            },
            "quest_data": {
                "available_quests": [
                    {
                        "id": "deliver_ore_005",
                        "prerequisites": ["gather_ore_003"],
                        "level_req": 10,
                    },
                ],
            },
            "relationship_data": {
                "player_npc_relationship": 35,
                "relationship_tier": "friendly",
            },
            "world_context": {
                "time_of_day": "afternoon",
                "weather": "clear",
                "active_events": [],
            },
        }

    def _get_production_state(self, player_id: str, npc_id: str) -> Dict[str, Any]:
        """从真实游戏数据库查询世界状态（生产用）"""
        if self._game_db_client is None:
            raise RuntimeError("Production DB client not configured")

        # 实际实现需要根据游戏后端API调整
        player_data = self._game_db_client.get_player(player_id)
        npc_data = self._game_db_client.get_npc(npc_id)
        relationship = self._game_db_client.get_relationship(player_id, npc_id)

        return {
            "player_info": player_data,
            "npc_inventory": npc_data.get("inventory", {}),
            "quest_data": self._game_db_client.get_available_quests(player_id, npc_id),
            "relationship_data": relationship,
            "world_context": self._game_db_client.get_world_context(),
        }

    @staticmethod
    def to_tool_return_expected(world_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        将世界状态转换为 tool_return_expected 格式。
        这是原始框架中mock_tools使用的格式。
        """
        return {
            "query_player_info": {
                "stats": {
                    "player_level": world_state["player_info"]["level"],
                    "player_class": world_state["player_info"]["class"],
                    "gold_balance": world_state["player_info"]["gold"],
                },
                "reputation": world_state["player_info"].get("faction_reputation", {}),
                "balance": {"gold_balance": world_state["player_info"]["gold"]},
            },
            "query_inventory": {
                "items": world_state["npc_inventory"].get("items", []),
                "total_slots": 20,
                "used_slots": len(world_state["npc_inventory"].get("items", [])),
            },
            "query_quest_status": world_state.get("quest_data", {}),
            "relationship": world_state.get("relationship_data", {}),
        }
