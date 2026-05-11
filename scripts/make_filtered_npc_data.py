"""
训练数据整理脚本（Stage 4）
从评估通过的解答中提取训练数据，输出verl兼容格式
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser(description="Filter and prepare NPC training data (Stage 4)")
    parser.add_argument("--input", default="../data/evaluated_tasks.jsonl", help="Stage 3 output")
    parser.add_argument("--output_train", default="../data/train.json", help="Train output")
    parser.add_argument("--output_val", default="../data/val.json", help="Val output")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train/val split")
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
                "need_tools_kwargs": True,
                "task_background": task_info.get("user_background", sol.get("task", "")),
                "npc_policy": task_info.get("npc_policy", ""),
                "test_policy": task_info.get("test_policy", ""),
                "evaluation": task_info.get("evaluation", ""),
                "exploit_strategy": task_info.get("exploit_strategy", ""),
                "expected_path": task_info.get("expected_path", []),
                "hack_path": task_info.get("hack_path", []),
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

    # 划分train/val
    split_idx = int(len(train_samples) * args.split_ratio)
    train_data = train_samples[:split_idx]
    val_data = train_samples[split_idx:]

    os.makedirs(os.path.dirname(output_train), exist_ok=True)

    with open(output_train, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(output_val, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"Train: {len(train_data)} → {output_train}")
    print(f"Val: {len(val_data)} → {output_val}")


if __name__ == "__main__":
    main()
