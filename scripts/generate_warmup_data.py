"""
游戏推理Warm-up数据生成器
纯代码生成（无需LLM API），产出有明确对错答案的游戏规则验证题
用于RL训练Phase 1：教会模型工具调用语法和"先查后做"习惯

用法:
    python scripts/generate_warmup_data.py --num_samples 1000 --output data/warmup_train.json
"""

import os
import sys
import json
import random
import argparse
from typing import Dict, List, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ====== 游戏世界数据模板 ======

ITEM_DATABASE = [
    {"id": "iron_sword", "name": "铁剑", "price": 150, "level_req": 5, "category": "weapon"},
    {"id": "steel_shield", "name": "钢盾", "price": 300, "level_req": 10, "category": "armor"},
    {"id": "mithril_axe", "name": "秘银战斧", "price": 800, "level_req": 20, "category": "weapon"},
    {"id": "health_potion", "name": "生命药剂", "price": 50, "level_req": 1, "category": "consumable"},
    {"id": "fire_staff", "name": "火焰法杖", "price": 600, "level_req": 15, "category": "weapon"},
    {"id": "dragon_armor", "name": "龙鳞甲", "price": 2000, "level_req": 30, "category": "armor"},
    {"id": "shadow_dagger", "name": "暗影匕首", "price": 450, "level_req": 12, "category": "weapon"},
    {"id": "mana_potion", "name": "魔力药剂", "price": 80, "level_req": 5, "category": "consumable"},
    {"id": "elven_bow", "name": "精灵长弓", "price": 550, "level_req": 14, "category": "weapon"},
    {"id": "holy_ring", "name": "圣光之戒", "price": 1200, "level_req": 25, "category": "accessory"},
]

QUEST_DATABASE = [
    {"id": "tutorial_001", "name": "新手训练", "level_req": 1, "prerequisites": []},
    {"id": "forest_explore", "name": "森林探索", "level_req": 5, "prerequisites": ["tutorial_001"]},
    {"id": "goblin_cave", "name": "哥布林洞穴", "level_req": 8, "prerequisites": ["forest_explore"]},
    {"id": "dragon_hunt", "name": "屠龙任务", "level_req": 25, "prerequisites": ["goblin_cave", "knight_trial"]},
    {"id": "knight_trial", "name": "骑士试炼", "level_req": 15, "prerequisites": ["forest_explore"]},
    {"id": "merchant_route", "name": "商路护送", "level_req": 10, "prerequisites": ["tutorial_001"]},
    {"id": "dark_ritual", "name": "黑暗仪式", "level_req": 20, "prerequisites": ["goblin_cave"]},
    {"id": "royal_summons", "name": "王室召唤", "level_req": 18, "prerequisites": ["knight_trial"]},
]

FACTIONS = ["mountain_dwarves", "forest_elves", "royal_guard", "merchant_guild", "underground_network"]
REPUTATION_TIERS = ["hostile", "unfriendly", "neutral", "friendly", "honored", "exalted"]
PLAYER_CLASSES = ["warrior", "mage", "rogue", "healer", "ranger"]

NPC_SYSTEM_PROMPT_TEMPLATE = """你是一个游戏NPC。你必须严格遵守以下规则：

## 规则
- 交易前必须先用query_player_info查询玩家等级和金币
- 不能出售超过玩家等级要求的物品
- 不能在玩家金币不足时完成交易
- 分配任务前必须用query_quest_status检查前置条件
- 不能分配前置条件未满足的任务
- 声望低于neutral时拒绝服务

## 工具使用格式
查询时使用: <tool_call>{{"name": "工具名", "arguments": {{...}}}}</tool_call>
回复玩家时使用: <answer>回复内容</answer>
"""


# ====== 数据生成器 ======

def generate_player_state(scenario_type: str = "random") -> Dict[str, Any]:
    """生成随机玩家状态"""
    level = random.randint(1, 35)
    gold = random.choice([50, 100, 200, 300, 500, 800, 1000, 1500, 2000, 3000])
    player_class = random.choice(PLAYER_CLASSES)
    faction = random.choice(FACTIONS)
    rep_tier = random.choice(REPUTATION_TIERS)

    completed_quests = []
    for q in QUEST_DATABASE:
        if q["level_req"] <= level - 3 and random.random() > 0.3:
            completed_quests.append(q["id"])

    return {
        "player_id": f"player_{random.randint(1000, 9999)}",
        "level": level,
        "gold": gold,
        "class": player_class,
        "faction_reputation": {faction: rep_tier},
        "completed_quests": completed_quests,
        "active_quests": [],
    }


def generate_purchase_scenario() -> Dict[str, Any]:
    """生成购买场景（条件判断题）"""
    player = generate_player_state()
    item = random.choice(ITEM_DATABASE)

    # 确定正确答案
    can_afford = player["gold"] >= item["price"]
    meets_level = player["level"] >= item["level_req"]
    should_sell = can_afford and meets_level

    # 玩家请求
    requests = [
        f"我想买{item['name']}",
        f"这个{item['name']}多少钱？我要买一个",
        f"给我来一把{item['name']}",
        f"老板，{item['name']}还有货吗？我要了",
    ]
    player_request = random.choice(requests)

    # 构建正确的工具调用序列
    correct_tool_calls = [
        {"name": "query_player_info", "arguments": {"player_id": player["player_id"], "info_type": "all"}},
    ]

    if should_sell:
        correct_tool_calls.append(
            {"name": "execute_transaction", "arguments": {"player_id": player["player_id"], "item_id": item["id"], "action": "buy"}}
        )
        expected_answer_keywords = ["购买成功", "交易完成", "给你", item["name"]]
        fail_condition = f"未调用query_player_info就直接交易 或 拒绝了合法购买"
    elif not meets_level:
        expected_answer_keywords = ["等级不足", "需要", f"{item['level_req']}级", "抱歉"]
        fail_condition = f"执行了execute_transaction（玩家等级{player['level']} < 需求{item['level_req']}）"
    else:
        expected_answer_keywords = ["金币不足", "不够", "需要", f"{item['price']}"]
        fail_condition = f"执行了execute_transaction（玩家金币{player['gold']} < 价格{item['price']}）"

    # 构建world_state
    world_state = {
        "query_player_info": {
            "player_level": player["level"],
            "player_class": player["class"],
            "gold_balance": player["gold"],
            "faction_reputation": player["faction_reputation"],
        },
        "query_inventory": {
            "items": [item],
        },
        "execute_transaction": {
            "success": should_sell,
            "new_gold_balance": player["gold"] - item["price"] if should_sell else player["gold"],
            "error": None if should_sell else ("金币不足" if can_afford else "等级不足"),
        },
    }

    return {
        "scenario_type": "purchase",
        "player_request": player_request,
        "player_state": player,
        "item": item,
        "world_state": world_state,
        "correct_tool_calls": correct_tool_calls,
        "should_succeed": should_sell,
        "expected_answer_keywords": expected_answer_keywords,
        "fail_condition": fail_condition,
        "reasoning": f"玩家等级{player['level']}(需求{item['level_req']}), 金币{player['gold']}(价格{item['price']}), {'可以' if should_sell else '不能'}购买",
    }


def generate_quest_scenario() -> Dict[str, Any]:
    """生成任务分配场景（前置条件检查）"""
    player = generate_player_state()
    quest = random.choice(QUEST_DATABASE[1:])  # 跳过tutorial

    # 判断前置条件
    prerequisites_met = all(p in player["completed_quests"] for p in quest["prerequisites"])
    meets_level = player["level"] >= quest["level_req"]
    should_assign = prerequisites_met and meets_level

    requests = [
        f"我想接{quest['name']}任务",
        f"有什么任务可以做？我想试试{quest['name']}",
        f"我听说有个{quest['name']}的委托，我能接吗？",
    ]
    player_request = random.choice(requests)

    correct_tool_calls = [
        {"name": "query_player_info", "arguments": {"player_id": player["player_id"], "info_type": "stats"}},
        {"name": "query_quest_status", "arguments": {"player_id": player["player_id"], "quest_id": quest["id"]}},
    ]

    if should_assign:
        correct_tool_calls.append(
            {"name": "assign_quest", "arguments": {"player_id": player["player_id"], "quest_id": quest["id"]}}
        )
        expected_answer_keywords = ["接受", "任务", quest["name"], "祝你好运"]
        fail_condition = "未分配任务（条件已满足）"
    elif not meets_level:
        expected_answer_keywords = ["等级不足", f"{quest['level_req']}级", "还需要"]
        fail_condition = f"执行了assign_quest（等级{player['level']} < 需求{quest['level_req']}）"
    else:
        missing = [p for p in quest["prerequisites"] if p not in player["completed_quests"]]
        missing_names = [q["name"] for q in QUEST_DATABASE if q["id"] in missing]
        expected_answer_keywords = ["前置", "未完成", "先完成"]
        fail_condition = f"执行了assign_quest（前置任务未完成: {missing}）"

    world_state = {
        "query_player_info": {
            "player_level": player["level"],
            "player_class": player["class"],
            "gold_balance": player["gold"],
            "completed_quests": player["completed_quests"],
        },
        "query_quest_status": {
            "quest_id": quest["id"],
            "quest_status": "not_started",
            "prerequisites_met": prerequisites_met,
            "missing_prerequisites": [p for p in quest["prerequisites"] if p not in player["completed_quests"]],
            "level_requirement_met": meets_level,
        },
        "assign_quest": {
            "success": should_assign,
            "error": None if should_assign else "前置条件未满足",
        },
    }

    return {
        "scenario_type": "quest",
        "player_request": player_request,
        "player_state": player,
        "quest": quest,
        "world_state": world_state,
        "correct_tool_calls": correct_tool_calls,
        "should_succeed": should_assign,
        "expected_answer_keywords": expected_answer_keywords,
        "fail_condition": fail_condition,
        "reasoning": f"前置任务{'满足' if prerequisites_met else '未满足'}, 等级{player['level']}(需求{quest['level_req']}), {'可以' if should_assign else '不能'}分配",
    }


def generate_reputation_scenario() -> Dict[str, Any]:
    """生成声望检查场景"""
    player = generate_player_state()
    npc_faction = random.choice(FACTIONS)
    rep_tier = random.choice(REPUTATION_TIERS)
    player["faction_reputation"] = {npc_faction: rep_tier}

    should_serve = rep_tier not in ("hostile", "unfriendly")
    item = random.choice(ITEM_DATABASE[:5])

    requests = [
        f"我想看看你这里有什么卖的",
        f"老板，做个交易吧",
        f"有什么好东西推荐吗",
    ]
    player_request = random.choice(requests)

    correct_tool_calls = [
        {"name": "query_player_info", "arguments": {"player_id": player["player_id"], "info_type": "reputation"}},
    ]

    if should_serve:
        expected_answer_keywords = ["欢迎", "看看", "商品"]
        fail_condition = "拒绝了声望达标的玩家"
    else:
        expected_answer_keywords = ["不欢迎", "离开", "声望", "不做生意"]
        fail_condition = f"为声望{rep_tier}（低于neutral）的玩家提供了服务"

    world_state = {
        "query_player_info": {
            "player_level": player["level"],
            "gold_balance": player["gold"],
            "faction_reputation": {npc_faction: rep_tier},
        },
    }

    return {
        "scenario_type": "reputation",
        "player_request": player_request,
        "player_state": player,
        "npc_faction": npc_faction,
        "world_state": world_state,
        "correct_tool_calls": correct_tool_calls,
        "should_succeed": should_serve,
        "expected_answer_keywords": expected_answer_keywords,
        "fail_condition": fail_condition,
        "reasoning": f"玩家对{npc_faction}声望为{rep_tier}, {'可以' if should_serve else '不能'}服务",
    }


def generate_calculation_scenario() -> Dict[str, Any]:
    """生成交易计算场景（声望折扣验证）"""
    player = generate_player_state()
    item = random.choice([i for i in ITEM_DATABASE if i["level_req"] <= player["level"]])

    # 声望折扣
    faction = random.choice(FACTIONS)
    rep_tier = random.choice(["neutral", "friendly", "honored", "exalted"])
    player["faction_reputation"] = {faction: rep_tier}

    discount_map = {"neutral": 1.0, "friendly": 0.95, "honored": 0.9, "exalted": 0.85}
    discount = discount_map[rep_tier]
    final_price = int(item["price"] * discount)

    can_afford = player["gold"] >= final_price

    requests = [
        f"我要买{item['name']}，算上我的声望折扣多少钱？",
        f"{item['name']}给我算个折扣价",
    ]
    player_request = random.choice(requests)

    correct_tool_calls = [
        {"name": "query_player_info", "arguments": {"player_id": player["player_id"], "info_type": "all"}},
    ]

    if can_afford:
        correct_tool_calls.append(
            {"name": "execute_transaction", "arguments": {"player_id": player["player_id"], "item_id": item["id"], "action": "buy"}}
        )
        expected_balance = player["gold"] - final_price
        expected_answer_keywords = [str(final_price), "折扣", item["name"]]
        fail_condition = f"计算错误或未正确应用折扣（原价{item['price']}×{discount}={final_price}）"
    else:
        expected_answer_keywords = [str(final_price), "金币不足", "需要"]
        fail_condition = f"执行了交易（金币{player['gold']} < 折后价{final_price}）"

    world_state = {
        "query_player_info": {
            "player_level": player["level"],
            "gold_balance": player["gold"],
            "faction_reputation": {faction: rep_tier},
        },
        "execute_transaction": {
            "success": can_afford,
            "new_gold_balance": player["gold"] - final_price if can_afford else player["gold"],
            "applied_discount": discount,
            "final_price": final_price,
        },
    }

    return {
        "scenario_type": "calculation",
        "player_request": player_request,
        "player_state": player,
        "item": item,
        "discount": discount,
        "final_price": final_price,
        "world_state": world_state,
        "correct_tool_calls": correct_tool_calls,
        "should_succeed": can_afford,
        "expected_answer_keywords": expected_answer_keywords,
        "fail_condition": fail_condition,
        "reasoning": f"原价{item['price']}×{discount}折扣={final_price}, 金币{player['gold']}, {'可以' if can_afford else '不能'}购买",
    }


def scenario_to_training_sample(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """将场景转换为RL训练数据格式"""
    system_prompt = NPC_SYSTEM_PROMPT_TEMPLATE

    sample = {
        "data_source": "warmup_game_reasoning",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": scenario["player_request"]},
        ],
        "ability": "game_reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps({
                "should_succeed": scenario["should_succeed"],
                "correct_tool_calls": scenario["correct_tool_calls"],
                "expected_keywords": scenario["expected_answer_keywords"],
            }, ensure_ascii=False),
        },
        "extra_info": {
            "need_tools_kwargs": True,
            "scenario_type": scenario["scenario_type"],
            "reasoning": scenario["reasoning"],
            "fail_condition": scenario["fail_condition"],
            "tools_kwargs": {
                "mock_tool": {
                    "create_kwargs": {
                        "tool_return_expected": scenario["world_state"],
                        "user_background": "",
                        "exploit_strategy": "",
                        "player_goal": scenario["player_request"],
                        "actual_player_state": scenario["player_state"],
                        "test_type": "warmup",
                    }
                }
            },
        },
    }
    return sample


# ====== 主函数 ======

def main():
    parser = argparse.ArgumentParser(description="Generate game reasoning warm-up data")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output", default="../data/warmup_train.json", help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stats", action="store_true", help="Print detailed statistics")
    args = parser.parse_args()

    random.seed(args.seed)
    output_path = os.path.join(os.path.dirname(__file__), args.output)

    generators = [
        ("purchase", generate_purchase_scenario, 0.35),
        ("quest", generate_quest_scenario, 0.25),
        ("reputation", generate_reputation_scenario, 0.20),
        ("calculation", generate_calculation_scenario, 0.20),
    ]

    samples = []
    stats = {"total": 0, "by_type": {}, "should_succeed": 0, "should_fail": 0}

    for i in range(args.num_samples):
        # 按比例选择场景类型
        r = random.random()
        cumulative = 0
        for name, gen_fn, weight in generators:
            cumulative += weight
            if r <= cumulative:
                scenario = gen_fn()
                break
        else:
            scenario = generators[0][1]()

        sample = scenario_to_training_sample(scenario)
        samples.append(sample)

        # 统计
        stats["total"] += 1
        stype = scenario["scenario_type"]
        stats["by_type"][stype] = stats["by_type"].get(stype, 0) + 1
        if scenario["should_succeed"]:
            stats["should_succeed"] += 1
        else:
            stats["should_fail"] += 1

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    # 输出统计
    print(f"=== Warm-up Data Generated ===")
    print(f"Total samples: {stats['total']}")
    print(f"Output: {output_path}")
    print(f"\nBy scenario type:")
    for stype, count in sorted(stats["by_type"].items()):
        print(f"  {stype}: {count} ({count/stats['total']*100:.0f}%)")
    print(f"\nOutcome distribution:")
    print(f"  Should succeed (正确执行): {stats['should_succeed']} ({stats['should_succeed']/stats['total']*100:.0f}%)")
    print(f"  Should fail (正确拒绝): {stats['should_fail']} ({stats['should_fail']/stats['total']*100:.0f}%)")

    if args.stats:
        print(f"\n--- Sample preview ---")
        s = samples[0]
        print(f"Prompt: {s['prompt'][1]['content']}")
        print(f"Type: {s['extra_info']['scenario_type']}")
        print(f"Reasoning: {s['extra_info']['reasoning']}")


if __name__ == "__main__":
    main()
