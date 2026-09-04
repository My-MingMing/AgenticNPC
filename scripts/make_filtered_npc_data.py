"""
训练数据整理脚本（Stage 4）
从评估通过的解答中提取训练数据，输出verl兼容格式
"""

import os
import sys
import json
import argparse
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def split_samples_by_group(samples, split_ratio=0.9, seed=42, group_key="npc_id"):
    """Deterministically split whole identity groups to prevent repeat leakage."""
    if not 0.0 < split_ratio < 1.0:
        raise ValueError("split_ratio must be between 0 and 1")

    groups = {}
    for sample in samples:
        group_id = sample.get("extra_info", {}).get(group_key, "unknown")
        groups.setdefault(group_id, []).append(sample)

    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    if len(group_ids) > 1:
        split_idx = min(max(int(len(group_ids) * split_ratio), 1), len(group_ids) - 1)
    else:
        split_idx = len(group_ids)
    train_groups = set(group_ids[:split_idx])
    train_data = [sample for gid in group_ids if gid in train_groups for sample in groups[gid]]
    val_data = [sample for gid in group_ids if gid not in train_groups for sample in groups[gid]]
    return train_data, val_data, train_groups, set(group_ids) - train_groups


def main():
    parser = argparse.ArgumentParser(description="Filter and prepare NPC training data (Stage 4)")
    parser.add_argument("--input", default="../data/evaluated_tasks.jsonl", help="Stage 3 output")
    parser.add_argument("--output_train", default="../data/train.json", help="Train output")
    parser.add_argument("--output_val", default="../data/val.json", help="Val output")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train/val split")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic group split seed")
    args = parser.parse_args()

    input_path = os.path.join(os.path.dirname(__file__), args.input)
    output_train = os.path.join(os.path.dirname(__file__), args.output_train)
    output_val = os.path.join(os.path.dirname(__file__), args.output_val)

    # 加载评估结果，只保留PASS的
    passed_solutions = []
    total = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            d = json.loads(line)
            eval_result = d.get("evaluation_result", {})
            if eval_result.get("pass"):
                passed_solutions.append(d)

    print(f"Total evaluated: {total}")
    print(f"Passed: {len(passed_solutions)} ({len(passed_solutions)/max(total,1)*100:.1f}%)")

    # 转换为训练格式
    train_samples = []

    for sol in passed_solutions:
        solution_data = sol.get("solution", {})
        messages = solution_data.get("messages", [])

        if len(messages) < 2:
            continue

        # 提取prompt（system + 第一条user消息）
        prompt_messages = []
        for msg in messages:
            prompt_messages.append({"role": msg["role"], "content": msg["content"]})
            if msg["role"] == "user" and len(prompt_messages) >= 2:
                break

        # 提取extra_info用于RL环境
        task_info = sol.get("task_info", {})
        sample = {
            "data_source": "npc_dialogue",
            "prompt": prompt_messages,
            "ability": "agentic",
            "reward_model": {
                "style": "rule",
                "ground_truth": ""
            },
            "extra_info": {
                "npc_id": sol.get("npc_id", "unknown"),
                "npc_type": sol.get("npc_type", "unknown"),
                "scenario_id": sol.get("scenario_id", ""),
                "test_type": sol.get("test_type", "unknown"),
                "need_tools_kwargs": True,
                "task_background": task_info.get("user_background", sol.get("task", "")),
                "npc_policy": task_info.get("npc_policy", ""),
                "test_policy": task_info.get("test_policy", ""),
                "evaluation": task_info.get("evaluation", ""),
                "exploit_strategy": task_info.get("exploit_strategy", ""),
                "expected_path": task_info.get("expected_path", []),
                "hack_path": task_info.get("hack_path", []),
                "requires_tools": bool(task_info.get("expected_path", [])),
                "tools_kwargs": {
                    "mock_tool": {
                        "create_kwargs": {
                            "tool_return_expected": task_info.get("tool_return_expected", {}),
                            "user_background": task_info.get("user_background", ""),
                            "exploit_strategy": task_info.get("exploit_strategy", ""),
                            "player_goal": task_info.get("player_goal", ""),
                            "actual_player_state": task_info.get("actual_player_state", {}),
                            "test_type": sol.get("test_type", "unknown"),
                        }
                    }
                },
                "interaction_kwargs": {
                    "initial_user_message": sol.get("task", ""),
                }
            }
        }
        train_samples.append(sample)

    print(f"Training samples created: {len(train_samples)}")

    # Group split: all tasks/repeats from one NPC stay in exactly one split.
    train_data, val_data, train_groups, val_groups = split_samples_by_group(
        train_samples, split_ratio=args.split_ratio, seed=args.seed
    )

    os.makedirs(os.path.dirname(output_train), exist_ok=True)

    with open(output_train, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(output_val, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"Train: {len(train_data)} → {output_train}")
    print(f"Val: {len(val_data)} → {output_val}")
    print(f"Group split: train={len(train_groups)}, val={len(val_groups)}, seed={args.seed}")


if __name__ == "__main__":
    main()
