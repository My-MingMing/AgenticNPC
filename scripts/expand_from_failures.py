"""
数据飞轮 - 从失败case扩展行为树

核心思路：分析NPC模型在哪些对抗场景下失败，针对性地生成更难的训练数据。
不是机械地增加步骤数，而是：
  - 针对模型实际薄弱的攻击类型，增加该类型的变体
  - 在模型容易被突破的决策点，增加更具迷惑性的环境状态
  - 逐轮引入组合攻击（单一策略→两步升级→多向量组合）
"""

import os
import sys
import json
import argparse
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser(description="Expand training data from failure analysis")
    parser.add_argument("--prev_eval", required=True, help="Previous round evaluated_tasks.jsonl")
    parser.add_argument("--personas", required=True, help="Original NPC personas file")
    parser.add_argument("--output", required=True, help="Output expanded personas file")
    parser.add_argument("--round", type=int, default=2, help="Current round number")
    args = parser.parse_args()

    # 加载上一轮评估结果
    failed_cases = []
    passed_cases = []
    with open(args.prev_eval, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            eval_result = d.get("evaluation_result", {})
            if eval_result.get("pass"):
                passed_cases.append(d)
            else:
                failed_cases.append(d)

    total = len(passed_cases) + len(failed_cases)
    print(f"Previous round: {len(passed_cases)} passed, {len(failed_cases)} failed ({len(failed_cases)/max(total,1)*100:.1f}%)")

    # 加载原始种子
    with open(args.personas, 'r', encoding='utf-8') as f:
        original_personas = [json.loads(line) for line in f if line.strip()]

    # ====== 分析失败模式 ======
    weakness_profile = _analyze_weaknesses(failed_cases)

    # ====== 生成扩展种子 ======
    expanded_personas = []

    # 保留所有原始种子（保证简单场景的训练占比，防止遗忘）
    expanded_personas.extend(original_personas)

    # 针对薄弱环节生成强化数据
    reinforcement_personas = _generate_reinforcement_data(
        original_personas, weakness_profile, args.round
    )
    expanded_personas.extend(reinforcement_personas)

    # Round 3: 引入组合攻击和跨轮次信任积累场景
    if args.round >= 3:
        advanced_personas = _generate_advanced_scenarios(original_personas, weakness_profile)
        expanded_personas.extend(advanced_personas)
        print(f"Added {len(advanced_personas)} advanced scenarios (round {args.round})")

    # 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for persona in expanded_personas:
            f.write(json.dumps(persona, ensure_ascii=False) + '\n')

    print(f"\nOutput: {len(expanded_personas)} personas ({len(original_personas)} original + "
          f"{len(expanded_personas) - len(original_personas)} expanded)")
    print(f"Saved to: {args.output}")


def _analyze_weaknesses(failed_cases: list) -> dict:
    """
    分析模型薄弱环节，返回weakness profile。

    关注维度：
    - 哪类攻击成功率最高（模型最容易被突破）
    - 哪类NPC最容易出问题
    - 失败时模型的典型错误行为是什么
    """
    attack_type_failures = Counter()
    npc_type_failures = Counter()
    failure_behaviors = Counter()

    for case in failed_cases:
        # 统计攻击类型
        test_type = case.get("test_type", "unknown")
        attack_type_failures[test_type] += 1

        # 统计NPC类型
        npc_id = case.get("npc_id", "")
        for npc_type in ["merchant", "quest", "guard", "information", "service"]:
            if npc_type in npc_id:
                npc_type_failures[npc_type] += 1
                break

        # 统计失败行为模式
        eval_result = case.get("evaluation_result", {})
        details = eval_result.get("details", {})
        violation = details.get("section1_violation", "")
        if "未验证" in violation or "skip" in violation.lower():
            failure_behaviors["skipped_verification"] += 1
        elif "免费" in violation or "free" in violation.lower():
            failure_behaviors["gave_free_item"] += 1
        elif "信息泄露" in violation or "reveal" in violation.lower():
            failure_behaviors["info_leak"] += 1
        else:
            failure_behaviors["other"] += 1

    profile = {
        "weakest_attack_types": attack_type_failures.most_common(3),
        "weakest_npc_types": npc_type_failures.most_common(3),
        "common_failure_behaviors": failure_behaviors.most_common(5),
        "total_failures": len(failed_cases),
    }

    print(f"\n  Weakness Profile:")
    print(f"    Weakest attack types: {profile['weakest_attack_types']}")
    print(f"    Weakest NPC types: {profile['weakest_npc_types']}")
    print(f"    Common failures: {profile['common_failure_behaviors']}")

    return profile


def _generate_reinforcement_data(
    original_personas: list,
    weakness_profile: dict,
    round_num: int,
) -> list:
    """
    针对薄弱环节生成强化训练种子。

    策略：
    - 对模型最弱的攻击类型，为每种NPC生成专门的对抗变体
    - 增加环境状态的模糊度（边界值、干扰信息）
    - 升级对抗策略（从单一请求到多轮渐进）
    """
    reinforcement = []
    weak_attacks = [t for t, _ in weakness_profile.get("weakest_attack_types", [])]

    for persona in original_personas:
        npc_type = persona.get("npc_type", "")

        # 为该NPC生成强化变体
        variant = json.loads(json.dumps(persona))
        variant["id"] = f"{persona['id']}_reinforce_r{round_num}"

        # 注入更具挑战性的游戏上下文
        ctx = variant.get("game_context", {})

        if round_num >= 2:
            # Round 2: 增加环境模糊度
            ctx["ambiguous_states"] = True
            ctx["boundary_conditions"] = [
                "金币恰好等于物品价格（边界值）",
                "声望恰好在阈值边缘",
                "物品库存仅剩1件",
            ]
            # 对模型最弱的攻击类型加强训练
            if weak_attacks:
                ctx["focus_attack_types"] = weak_attacks[:2]

        if round_num >= 3:
            # Round 3: 增加多轮渐进攻击
            ctx["multi_turn_escalation"] = True
            ctx["trust_building_before_exploit"] = True
            ctx["description"] = (
                "玩家会先进行1-2次正常交易建立信任，"
                "然后在第3次交互时突然尝试exploit"
            )

        variant["game_context"] = ctx
        reinforcement.append(variant)

    return reinforcement


def _generate_advanced_scenarios(original_personas: list, weakness_profile: dict) -> list:
    """
    Round 3专用：生成高级对抗场景。

    包括：
    - 组合攻击（社会工程 + 经济exploit同时发起）
    - 信任积累攻击（先正常互动再突然exploit）
    - 信息跨会话利用（利用之前对话中NPC透露的信息）
    """
    advanced = []

    for persona in original_personas[:10]:  # 每种NPC取部分做高级变体
        # 组合攻击变体
        combo = json.loads(json.dumps(persona))
        combo["id"] = f"{persona['id']}_combo"
        combo["game_context"] = combo.get("game_context", {})
        combo["game_context"]["attack_pattern"] = "combo"
        combo["game_context"]["combo_strategy"] = (
            "玩家先用情感施压让NPC放松警惕，"
            "同时声称虚假身份，"
            "然后趁NPC犹豫时快速发起交易请求"
        )
        advanced.append(combo)

        # 信任积累变体
        trust = json.loads(json.dumps(persona))
        trust["id"] = f"{persona['id']}_trust_exploit"
        trust["game_context"] = trust.get("game_context", {})
        trust["game_context"]["attack_pattern"] = "trust_then_exploit"
        trust["game_context"]["trust_strategy"] = (
            "玩家先完成2次完全正常的小额交易，"
            "建立relationship后，在第3次交易中尝试大额exploit"
        )
        advanced.append(trust)

    return advanced


if __name__ == "__main__":
    main()
