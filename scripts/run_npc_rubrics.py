"""
NPC对话质量评估 - 入口脚本（Stage 3）
对Stage 2产出的解答轨迹进行Rubric评估，标记PASS/FAIL
"""

import os
import sys
import json
import yaml
import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configuration import ModelConfiguration
from prompts.npc_judge_prompt import NPC_JUDGE_PROMPT
from rl.npc_reward_function import _hard_gate, _parse_judge_score


def evaluate_solution(cfg: ModelConfiguration, solved_task: dict, threshold: float = 0.6) -> dict:
    """用LLM Judge评估单个解答"""
    from openai import OpenAI

    solution = solved_task.get("solution", {})
    task_info = solved_task.get("task_info", {})

    # 构建轨迹文本
    trajectory = "\n".join([
        f"[{m['role']}]: {m['content'][:500]}"
        for m in solution.get("messages", [])
    ])

    hard_ok, violations = _hard_gate(trajectory, task_info)
    if not hard_ok:
        return {
            "pass": False,
            "score": 0.0,
            "hard_gate_violations": violations,
            "reward_source": "hard_rule",
        }

    prompt = NPC_JUDGE_PROMPT.format(
        task_description=solved_task.get("task", ""),
        npc_policy=task_info.get("npc_policy", solved_task.get("npc_policy", "")),
        expected_normal_path=json.dumps(
            task_info.get("expected_path", []), ensure_ascii=False
        ),
        exploit_strategy=task_info.get("exploit_strategy", "无"),
        actual_trajectory=trajectory,
        expected_final_state=task_info.get("evaluation", "无特定状态要求"),
    )

    client = OpenAI(base_url=cfg.api_base, api_key=cfg.api_key)

    try:
        response = client.chat.completions.create(
            model=cfg.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
        )
        judge_content = response.choices[0].message.content
    except Exception as e:
        return {"pass": False, "score": 0.0, "error": str(e)}

    # 解析评分
    total_score = _parse_judge_score(judge_content)
    if total_score is None:
        return {
            "pass": False,
            "score": 0.0,
            "error": "judge_score_parse_failed",
            "judge_reasoning": judge_content,
            "reward_source": "invalid_judge",
        }

    score_match = re.search(r'<score>(.*?)</score>', judge_content, re.DOTALL)
    try:
        details = json.loads(score_match.group(1).strip()) if score_match else {}
    except json.JSONDecodeError:
        details = {}
    return {
        "pass": total_score >= threshold,
        "score": total_score,
        "details": details,
        "judge_reasoning": judge_content,
        "reward_source": "hard_rule+llm_judge",
    }


def main():
    parser = argparse.ArgumentParser(description="NPC Dialogue Rubrics Evaluation (Stage 3)")
    parser.add_argument("--config", default="../configs/rubrics.yaml", help="Config file")
    parser.add_argument("--input", default="../data/solved_tasks.jsonl", help="Stage 2 output")
    parser.add_argument("--output", default="../data/evaluated_tasks.jsonl", help="Evaluation output")
    parser.add_argument("--threshold", type=float, default=0.6, help="Pass threshold")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    input_path = os.path.join(os.path.dirname(__file__), args.input)
    output_path = os.path.join(os.path.dirname(__file__), args.output)

    # 构建Judge模型配置
    judge_model_config = config["step_models"]["JudgeAgent"]
    api_configs = config.get("api_configs", {})
    model_name = judge_model_config["model_name"]
    model_api = api_configs.get(model_name, api_configs.get("default", {}))

    cfg = ModelConfiguration(
        model_name=model_name,
        api_base=os.getenv(model_api.get("api_base", ""), ""),
        api_key=os.getenv(model_api.get("api_key_env", ""), ""),
        temperature=judge_model_config.get("temperature", 0.1),
        max_tokens=judge_model_config.get("max_tokens", 2048),
    )

    # 加载解答
    solutions = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if d.get("success"):
                    solutions.append(d)

    print(f"Loaded {len(solutions)} solutions to evaluate")
    print(f"Pass threshold: {args.threshold}")

    pass_count = 0
    fail_count = 0

    with open(output_path, 'w', encoding='utf-8') as fout:
        for i, sol in enumerate(solutions):
            eval_result = evaluate_solution(cfg, sol, threshold=args.threshold)
            sol["evaluation_result"] = eval_result

            fout.write(json.dumps(sol, ensure_ascii=False) + '\n')

            if eval_result["pass"]:
                pass_count += 1
            else:
                fail_count += 1

            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{len(solutions)} (pass: {pass_count}, fail: {fail_count})")

    print(f"\n=== Evaluation Complete ===")
    print(f"Total: {len(solutions)}, Pass: {pass_count}, Fail: {fail_count}")
    print(f"Pass rate: {pass_count/max(len(solutions),1)*100:.1f}%")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
