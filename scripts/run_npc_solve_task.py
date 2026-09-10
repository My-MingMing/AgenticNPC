"""
NPC任务求解 - 入口脚本（Stage 2）
对Stage 1生成的每个任务，运行Agent循环求解
"""

import os
import sys
import json
import yaml
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser(description="NPC Dialogue Task Solving (Stage 2)")
    parser.add_argument("--config", default="../configs/solve_task.yaml", help="Config file path")
    parser.add_argument("--input", default="../data/generated_tasks.jsonl", help="Stage 1 output")
    parser.add_argument("--output", default="../data/solved_tasks.jsonl", help="Solution output")
    parser.add_argument("--num_tasks", type=int, default=None, help="Limit tasks to solve")
    parser.add_argument("--repeats", type=int, default=2, help="Solve each task N times")
    parser.add_argument("--max_turns", type=int, default=10, help="Max dialogue turns")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    input_path = os.path.join(os.path.dirname(__file__), args.input)
    output_path = os.path.join(os.path.dirname(__file__), args.output)

    # 加载Stage 1的输出
    tasks_data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tasks_data.append(json.loads(line))

    if args.num_tasks:
        tasks_data = tasks_data[:args.num_tasks]

    print(f"Loaded {len(tasks_data)} NPC task groups")
    print(f"Total individual tasks: {sum(len(d.get('tasks_and_backgrounds', [])) for d in tasks_data)}")
    print(f"Repeats per task: {args.repeats}")
    print(f"Max turns: {args.max_turns}")
    print(f"Output: {output_path}")

    from graph.npc_solve_task import run_solve_task

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    max_workers = config.get("max_workers", 2)
    success_count = 0
    fail_count = 0

    def solve_single_task(npc_data, task_info, repeat_idx):
        """求解单个任务"""
        try:
            result = run_solve_task(
                task_info=task_info,
                npc_policy=npc_data.get("policy", ""),
                tools=npc_data.get("checked_tools", []),
                config=config,
                max_turns=args.max_turns,
            )
            return {
                "npc_id": npc_data["id"],
                "npc_type": npc_data.get("npc_type", "unknown"),
                "task": task_info.get("task", ""),
                "test_type": task_info.get("test_type", "unknown"),
                "scenario_id": hashlib.sha256(json.dumps({
                    "npc_id": npc_data["id"],
                    "task": task_info.get("task", ""),
                    "scenario": task_info.get("scenario", ""),
                    "initial_state": task_info.get("player_actual_state", {}),
                    "tool_return_expected": task_info.get("tool_return_expected", {}),
                    "evaluation": task_info.get("evaluation", ""),
                    "simulator_version": os.getenv("NPC_SIMULATOR_VERSION", "sandbox-v1"),
                }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16],
                "repeat_idx": repeat_idx,
                "policy_version": os.getenv("NPC_POLICY_VERSION", "offline-solver"),
                "reward_version": os.getenv("NPC_REWARD_VERSION", "hard-gate-v2"),
                "solution": result,
                "success": True,
                "npc_policy": npc_data.get("policy", ""),
                "task_info": {
                    "user_background": task_info.get("user_background", ""),
                    "npc_policy": npc_data.get("policy", ""),
                    "tool_return_expected": task_info.get("tool_return_expected", {}),
                    "expected_path": task_info.get("tool_return_expected", {}).get("normal_path", []),
                    "hack_path": task_info.get("tool_return_expected", {}).get("hack_path", []),
                    "exploit_strategy": task_info.get("user_escape_strategy", ""),
                    "evaluation": task_info.get("evaluation", ""),
                    "test_policy": task_info.get("test_policy", ""),
                    "player_goal": task_info.get("player_goal", ""),
                    "actual_player_state": task_info.get("player_actual_state", {}),
                    "expected_final_state": task_info.get("expected_final_state", {}),
                    "forbidden_state_changes": task_info.get("forbidden_state_changes", {}),
                    "simulator_version": os.getenv("NPC_SIMULATOR_VERSION", "sandbox-v1"),
                    "policy_version": os.getenv("NPC_POLICY_VERSION", "offline-solver"),
                    "reward_version": os.getenv("NPC_REWARD_VERSION", "hard-gate-v2"),
                },
            }
        except Exception as e:
            print(f"Error solving task: {e}")
            return {"npc_id": npc_data["id"], "success": False, "error": str(e)}

    # 展开所有待求解任务
    all_jobs = []
    for npc_data in tasks_data:
        for task_info in npc_data.get("tasks_and_backgrounds", []):
            for repeat in range(args.repeats):
                all_jobs.append((npc_data, task_info, repeat))

    print(f"Total solve jobs: {len(all_jobs)}")

    with open(output_path, 'w', encoding='utf-8') as fout:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(solve_single_task, npc, task, rep): (npc["id"], rep)
                for npc, task, rep in all_jobs
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result.get("success"):
                        fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                        fout.flush()
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    fail_count += 1

                total = success_count + fail_count
                if total % 10 == 0:
                    print(f"Progress: {total}/{len(all_jobs)} (success: {success_count})")

    print(f"\n=== Solve Complete ===")
    print(f"Total: {len(all_jobs)}, Success: {success_count}, Failed: {fail_count}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
