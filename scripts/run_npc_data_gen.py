"""
NPC对话数据合成 - 入口脚本
用法: python run_npc_data_gen.py [--num_seeds N] [--dry_run]
"""

import os
import sys
import json
import yaml
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser(description="NPC Dialogue Data Synthesis")
    parser.add_argument("--config", default="../configs/data_gen.yaml", help="Config file path")
    parser.add_argument("--personas", default="../configs/npc_personas.jsonl", help="NPC personas file")
    parser.add_argument("--output", default=None, help="Override output path (default: from config)")
    parser.add_argument("--num_seeds", type=int, default=None, help="Limit number of seeds to process")
    parser.add_argument("--dry_run", action="store_true", help="Print config and exit")
    args = parser.parse_args()

    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 如果命令行指定了output，覆盖config中的路径
    if args.output:
        config["logging"]["task_file_path"] = args.output

    # 加载NPC种子
    personas_path = os.path.join(os.path.dirname(__file__), args.personas)
    with open(personas_path, 'r', encoding='utf-8') as f:
        personas = [json.loads(line) for line in f if line.strip()]

    if args.num_seeds:
        personas = personas[:args.num_seeds]

    print(f"Loaded {len(personas)} NPC personas")
    print(f"Output: {config['logging']['task_file_path']}")

    if args.dry_run:
        print("\n--- Dry Run ---")
        print(f"Config: {json.dumps(config, indent=2, ensure_ascii=False)}")
        print(f"\nFirst persona: {json.dumps(personas[0], indent=2, ensure_ascii=False)}")
        return

    # 运行数据合成
    from graph.npc_data_gen import run_npc_data_gen

    max_workers = config.get("max_workers", 4)
    success_count = 0
    fail_count = 0

    def process_persona(persona):
        seed = {
            "id": persona["id"],
            "persona": persona["persona"],
            "npc_type": persona.get("npc_type", "generic"),
            "normal_workflow": persona.get("normal_workflow"),
            "game_context": persona.get("game_context", {}),
        }
        try:
            result = run_npc_data_gen(seed, run_config=config)
            if result.get("breaked", True):
                # 打印失败阶段信息
                if result.get("npc_task") is None:
                    print(f"  [{persona['id']}] Failed at Stage 1 (behavior tree gen)")
                elif result.get("test_cases") is None:
                    print(f"  [{persona['id']}] Failed at Stage 2 (test case gen)")
                elif result.get("tasks_and_backgrounds") is None:
                    print(f"  [{persona['id']}] Failed at Stage 3 (refine)")
                return False
            return True
        except Exception as e:
            print(f"Error processing {persona['id']}: {e}")
            return False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_persona, p): p for p in personas}
        for future in as_completed(futures):
            persona = futures[future]
            try:
                success = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"Exception for {persona['id']}: {e}")
                fail_count += 1

            total = success_count + fail_count
            if total % 10 == 0:
                print(f"Progress: {total}/{len(personas)} (success: {success_count}, fail: {fail_count})")

    print(f"\n=== Complete ===")
    print(f"Total: {len(personas)}, Success: {success_count}, Failed: {fail_count}")
    print(f"Output saved to: {config['logging']['task_file_path']}")


if __name__ == "__main__":
    main()
