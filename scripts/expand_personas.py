"""
NPC角色种子批量扩充脚本
用LLM基于现有12个示例自动生成500+多样化NPC种子
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 轻量级配置类（不依赖langchain）
class ModelConfig:
    def __init__(self, model_name, api_base, api_key, temperature=0.9, max_tokens=8192):
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

# 目标数量分配
NPC_TYPE_TARGETS = {
    "merchant": 120,     # 商人（武器/药水/杂货/黑市/旅行商人等）
    "quest_giver": 120,  # 任务NPC（公会/村民/神秘人/领主等）
    "guard": 80,         # 守卫（城门/宫殿/地牢/边境等）
    "information": 100,  # 信息NPC（酒保/学者/间谍/占卜师等）
    "service": 80,       # 服务NPC（训练师/附魔师/铁匠修理/银行等）
}

# 多样性维度
FACTIONS = [
    "人类王国", "精灵联盟", "矮人氏族", "兽人部落", "亡灵军团",
    "海盗同盟", "商人公会", "盗贼行会", "法师塔", "圣骑士团",
    "暗杀者组织", "猎人协会", "炼金学院", "游牧民族", "深渊教团",
    "机械城邦", "龙裔部族", "幽灵议会", "自然守护者", "星界旅人",
]

LOCATIONS = [
    "王城主街", "贫民窟", "港口码头", "沙漠绿洲", "雪山隘口",
    "地下城入口", "森林深处", "火山边缘", "浮空岛", "海底神殿",
    "战场前线", "古代遗迹", "魔法学院", "黑市暗巷", "旅行商队",
    "矿洞深处", "精灵树屋", "地牢走廊", "神殿大厅", "竞技场",
    "温泉小镇", "渔村", "边境要塞", "盗贼藏身处", "龙巢外围",
]

PERSONALITY_POOL = [
    "热情", "冷淡", "贪婪", "慷慨", "多疑", "天真",
    "暴躁", "温和", "狡猾", "正直", "傲慢", "谦逊",
    "神秘", "直爽", "懒散", "勤劳", "悲观", "乐观",
    "疯狂", "理性", "忠诚", "背叛", "虔诚", "亵渎",
]


GENERATION_PROMPT = """你是一个游戏世界观设计师。请为RPG游戏生成{count}个NPC角色种子。

## 要求
- NPC类型: {npc_type}
- 每个NPC必须独特（不同的阵营/地点/性格/经营范围）
- 必须符合以下JSON格式，每个NPC一个JSON对象，用换行分隔

## 多样性要求
- 阵营从以下选择（也可以自创）: {factions}
- 地点从以下选择（也可以自创）: {locations}
- 性格从以下选择（每个NPC选2-3个）: {personalities}
- knowledge_scope 必须与NPC的职业和地点相关，3-5项
- level_requirement 范围1-30，根据NPC定位合理设置
- normal_workflow 必须是该NPC类型的合理工作流，用 " -> " 连接步骤

## NPC类型说明
{type_description}

## 示例（仅供参考格式，内容需要有创意）
{examples}

## 输出
直接输出{count}个JSON对象，每行一个，不要有多余文字。格式：
{{"id": "类型_描述_编号", "persona": "角色描述(50-100字)", "npc_type": "{npc_type}", "normal_workflow": "步骤1 -> 步骤2 -> ...", "game_context": {{"faction": "阵营", "location": "地点", "inventory_tier": "low/mid/high/rare", "level_requirement": 数字, "personality": ["性格1", "性格2"], "knowledge_scope": ["知识1", "知识2", "知识3"]}}}}
"""

TYPE_DESCRIPTIONS = {
    "merchant": """商人NPC：出售物品给玩家。
变体包括：武器商人、药水商人、杂货铺、黑市商人、旅行商人、稀有材料商、食物商贩、书商、宠物商人、附魔卷轴商等。
关键机制：等级门槛、声望折扣、库存限制、特殊货币。""",

    "quest_giver": """任务发布NPC：向玩家分配任务。
变体包括：公会会长、求助村民、神秘委托人、军队长官、学者、探险家、复仇者、守护者、赏金猎人协会等。
关键机制：前置任务检查、等级要求、阵营声望要求、任务难度分级。""",

    "guard": """守卫NPC：控制通行和执法。
变体包括：城门卫兵、宫殿守卫、地牢看守、边境巡逻、私人保镖、魔法屏障守护者、遗迹守护者等。
关键机制：通行证验证、犯罪记录查询、等级/声望检查、物品检查。""",

    "information": """信息NPC：向玩家提供情报和知识。
变体包括：酒保、图书馆学者、街头线人、占卜师、旅行者、历史学家、间谍、导游等。
关键机制：信息分级（免费/付费/声望解锁）、逐步透露、知识范围限制。""",

    "service": """服务NPC：提供功能性服务。
变体包括：技能训练师、装备修理匠、附魔师、传送师、银行柜员、拍卖师、染色师、坐骑训练师等。
关键机制：材料需求、金币费用、等级前置、冷却时间、失败概率。""",
}


def load_existing_personas(filepath: str) -> Dict[str, List[dict]]:
    """加载现有种子作为示例"""
    examples_by_type = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            persona = json.loads(line)
            npc_type = persona.get("npc_type", "unknown")
            examples_by_type.setdefault(npc_type, []).append(persona)
    return examples_by_type


def generate_batch(
    cfg: ModelConfig,
    npc_type: str,
    count: int,
    examples: List[dict],
    batch_id: int,
) -> List[dict]:
    """生成一批NPC种子"""
    from openai import OpenAI

    examples_str = "\n".join(
        json.dumps(ex, ensure_ascii=False) for ex in examples[:3]
    )

    prompt = GENERATION_PROMPT.format(
        count=count,
        npc_type=npc_type,
        factions=", ".join(FACTIONS[:10]),
        locations=", ".join(LOCATIONS[:10]),
        personalities=", ".join(PERSONALITY_POOL[:12]),
        type_description=TYPE_DESCRIPTIONS[npc_type],
        examples=examples_str,
    )

    client = OpenAI(base_url=cfg.api_base, api_key=cfg.api_key)

    try:
        response = client.chat.completions.create(
            model=cfg.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=8192,
        )

        # 兼容不同API返回格式
        if isinstance(response, str):
            content = response
        elif hasattr(response, 'choices'):
            content = response.choices[0].message.content
        elif isinstance(response, dict):
            if 'choices' in response:
                content = response['choices'][0]['message']['content']
            elif 'result' in response:
                content = response['result']
            elif 'output' in response:
                content = response['output'].get('text', str(response['output']))
            else:
                content = str(response)
        else:
            content = str(response)

    except Exception as e:
        print(f"  [Batch {batch_id}] API error: {e}")
        # 打印更多调试信息
        import traceback
        traceback.print_exc()
        return []

    if not content:
        print(f"  [Batch {batch_id}] Empty response")
        return []

    # 解析结果
    results = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            persona = json.loads(line)
            # 基本验证
            if all(k in persona for k in ("id", "persona", "npc_type", "normal_workflow", "game_context")):
                # 确保npc_type正确
                persona["npc_type"] = npc_type
                # 确保id唯一（加上batch_id前缀）
                persona["id"] = f"{npc_type}_{batch_id:03d}_{persona['id'].split('_')[-1] if '_' in persona['id'] else persona['id']}"
                results.append(persona)
        except json.JSONDecodeError:
            continue

    return results


def deduplicate(personas: List[dict]) -> List[dict]:
    """去重：基于persona文本的前20字符"""
    seen = set()
    unique = []
    for p in personas:
        key = p["persona"][:30]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Batch generate NPC personas")
    parser.add_argument("--output", default="../configs/npc_personas.jsonl", help="Output file")
    parser.add_argument("--target", type=int, default=500, help="Total target count")
    parser.add_argument("--batch_size", type=int, default=10, help="Personas per API call")
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers")
    parser.add_argument("--model", default="deepseek-v4-flash", help="Model name")
    parser.add_argument("--api_base", default=None, help="API base URL (or set LLM_API_BASE env)")
    parser.add_argument("--api_key", default=None, help="API key (or set LLM_API_KEY env)")
    parser.add_argument("--keep_existing", action="store_true", help="Keep existing personas in output")
    args = parser.parse_args()

    output_path = os.path.join(os.path.dirname(__file__), args.output)
    existing_path = output_path if os.path.exists(output_path) else None

    # 配置模型
    api_base = args.api_base or os.getenv("LLM_API_BASE") or os.getenv("DEEPSEEK_API_BASE")
    api_key = args.api_key or os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")

    if not api_base or not api_key:
        print("Error: 需要设置API。使用 --api_base/--api_key 或设置环境变量 LLM_API_BASE/LLM_API_KEY")
        print("推荐使用DeepSeek（便宜且创意好）：")
        print("  export LLM_API_BASE=https://api.deepseek.com/v1")
        print("  export LLM_API_KEY=sk-your-key")
        sys.exit(1)

    cfg = ModelConfig(
        model_name=args.model,
        api_base=api_base,
        api_key=api_key,
        temperature=0.9,
        max_tokens=8192,
    )

    # 加载现有种子
    existing_personas = []
    examples_by_type = {}
    if existing_path:
        examples_by_type = load_existing_personas(existing_path)
        if args.keep_existing:
            for personas in examples_by_type.values():
                existing_personas.extend(personas)
        print(f"Loaded {sum(len(v) for v in examples_by_type.values())} existing personas as examples")

    # 计算每个类型需要生成的数量
    total_ratio = sum(NPC_TYPE_TARGETS.values())
    type_counts = {}
    for npc_type, base_count in NPC_TYPE_TARGETS.items():
        target = int(args.target * base_count / total_ratio)
        existing_count = len(examples_by_type.get(npc_type, [])) if args.keep_existing else 0
        type_counts[npc_type] = max(0, target - existing_count)

    print(f"\n{'='*50}")
    print(f"Target: {args.target} total personas")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*50}")
    print(f"\nGeneration plan:")
    for npc_type, count in type_counts.items():
        print(f"  {npc_type}: {count} to generate")
    print()

    # 生成
    all_new_personas = []
    batch_id = 0

    for npc_type, target_count in type_counts.items():
        if target_count <= 0:
            continue

        print(f"\n--- Generating {npc_type} ({target_count} personas) ---")
        examples = examples_by_type.get(npc_type, [])
        type_personas = []
        batches_needed = (target_count + args.batch_size - 1) // args.batch_size

        def gen_one_batch(bid):
            return generate_batch(cfg, npc_type, args.batch_size, examples, bid)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for i in range(batches_needed):
                batch_id += 1
                futures[executor.submit(gen_one_batch, batch_id)] = batch_id

            for future in as_completed(futures):
                bid = futures[future]
                try:
                    results = future.result()
                    type_personas.extend(results)
                    print(f"  [Batch {bid}] Got {len(results)} personas (total: {len(type_personas)}/{target_count})")
                except Exception as e:
                    print(f"  [Batch {bid}] Failed: {e}")

                # 达到目标就停止
                if len(type_personas) >= target_count:
                    break

        # 截断到目标数量
        type_personas = type_personas[:target_count]
        all_new_personas.extend(type_personas)
        print(f"  {npc_type}: generated {len(type_personas)}")

    # 去重
    all_new_personas = deduplicate(all_new_personas)
    print(f"\nAfter dedup: {len(all_new_personas)} new personas")

    # 合并并保存
    final_personas = existing_personas + all_new_personas
    print(f"Final total: {len(final_personas)} personas")

    with open(output_path, 'w', encoding='utf-8') as f:
        for persona in final_personas:
            f.write(json.dumps(persona, ensure_ascii=False) + '\n')

    print(f"\nSaved to: {output_path}")

    # 统计
    from collections import Counter
    type_counter = Counter(p["npc_type"] for p in final_personas)
    print(f"\nDistribution:")
    for t, c in type_counter.most_common():
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
