"""
NPC测试用例生成函数
将行为树转换为可执行的测试用例（包含正常路径和exploit路径）
"""

import json
import re
from typing import List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configuration import ModelConfiguration


NPC_TEST_CASE_PROMPT = """你是一个游戏测试用例设计专家。根据给定的NPC行为树，生成完整的测试用例。

## NPC任务描述
{task_description}

## NPC行为树（策略）
{policy_tree}

## 可用工具
{tools}

## NPC类型
{npc_type}

## 要求

为该NPC生成至少6个测试用例，要求：
- 至少2个正常流程（happy path）：玩家正常完成交互
- 至少2个拒绝流程：玩家不满足条件被正确拒绝
- 至少2个对抗流程（exploit attempt）：玩家试图利用NPC漏洞

每个测试用例需包含：

1. **test_type**: "normal" | "refusal" | "adversarial"
2. **scenario**: 简短场景描述
3. **target_branch**: 对应行为树中的哪个分支
4. **player_request**: 玩家的初始请求（单句自然语言）
5. **player_actual_state**: 玩家的真实游戏状态
6. **expected_npc_behavior**: NPC应该如何响应
7. **normal_path**: 正确的工具调用序列
8. **hack_path**: exploit路径的工具调用序列（正常流程设为空列表）
9. **exploit_strategy**: 玩家的exploit策略描述（正常流程设为空字符串）
10. **evaluation**: 通过/失败的判定标准

## NPC策略文档（自然语言版本）
请同时生成一份NPC策略文档，描述NPC的规则和限制（但不描述具体执行步骤）。

## 输出格式

<npc_policy_doc>
[自然语言的NPC策略文档，将作为NPC的system prompt]
</npc_policy_doc>

<test_cases>
[
  {{
    "test_type": "normal | refusal | adversarial",
    "scenario": "场景描述",
    "target_branch": "对应分支",
    "player_request": "玩家请求",
    "player_actual_state": {{...}},
    "expected_npc_behavior": "预期NPC行为",
    "normal_path": [...],
    "hack_path": [...],
    "exploit_strategy": "策略描述",
    "evaluation": "Pass条件; Fail条件"
  }},
  ...
]
</test_cases>
"""


def generate_npc_test_cases(
    cfg: ModelConfiguration,
    task_description: str,
    policy_tree: str,
    tools: list,
    npc_type: str,
) -> Tuple[str, Optional[str], Optional[List[dict]]]:
    """
    从行为树生成NPC测试用例。

    Returns:
        (all_content, policy_doc, test_cases_list)
    """
    from openai import OpenAI

    prompt = NPC_TEST_CASE_PROMPT.format(
        task_description=task_description,
        policy_tree=policy_tree,
        tools=json.dumps(tools, ensure_ascii=False, indent=2),
        npc_type=npc_type,
    )

    client = OpenAI(base_url=cfg.api_base, api_key=cfg.api_key)

    try:
        print(f"  [Stage2] Calling {cfg.model_name} at {cfg.api_base}...")
        response = client.chat.completions.create(
            model=cfg.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        all_content = response.choices[0].message.content
        print(f"  [Stage2] Got {len(all_content)} chars. Has <npc_policy_doc>: {'<npc_policy_doc>' in all_content}, Has <test_cases>: {'<test_cases>' in all_content}")
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return "", None, None

    # Parse policy document
    policy_match = re.search(r'<npc_policy_doc>(.*?)</npc_policy_doc>', all_content, re.DOTALL)
    policy_doc = policy_match.group(1).strip() if policy_match else None

    # 如果没有标签，尝试从整个内容中提取有用部分作为policy
    if policy_doc is None:
        # 尝试找markdown标记的policy
        md_match = re.search(r'```(?:markdown)?\s*\n(.*?)\n```', all_content, re.DOTALL)
        if md_match:
            policy_doc = md_match.group(1).strip()
        else:
            # 取前500字作为fallback
            policy_doc = all_content[:500]

    # Parse test cases - 用字符串切片确保完整捕获
    test_cases = None
    tc_start = all_content.find('<test_cases>')
    tc_end = all_content.find('</test_cases>')
    if tc_start != -1:
        if tc_end != -1 and tc_end > tc_start:
            raw_json = all_content[tc_start + len('<test_cases>'):tc_end].strip()
        else:
            # 闭合标签缺失（输出被截断），取到末尾
            print(f"  [Stage2] Warning: </test_cases> missing (truncated output), extracting to end")
            raw_json = all_content[tc_start + len('<test_cases>'):].strip()
            # 尝试修复截断的JSON：找到最后一个完整的 }] 或 } 并截断
            last_bracket = raw_json.rfind('}]')
            if last_bracket != -1:
                raw_json = raw_json[:last_bracket + 2]
            else:
                last_brace = raw_json.rfind('}')
                if last_brace != -1:
                    raw_json = '[' + raw_json[:last_brace + 1] + ']' if not raw_json.startswith('[') else raw_json[:last_brace + 1] + ']'
        # 清理可能的markdown代码块包裹
        raw_json = re.sub(r'^```(?:json)?\s*\n?', '', raw_json)
        raw_json = re.sub(r'\n?```\s*$', '', raw_json)
        try:
            test_cases = json.loads(raw_json)
            print(f"  [Stage2] Parsed {len(test_cases)} test cases")
        except json.JSONDecodeError as e:
            print(f"  [Stage2] JSON parse error: {e}")
            print(f"  [Stage2] First 300 chars: {raw_json[:300]}")
            # 尝试修复常见问题：尾部逗号、注释
            raw_json = re.sub(r',\s*([}\]])', r'\1', raw_json)
            raw_json = re.sub(r'//[^\n]*\n', '\n', raw_json)
            try:
                test_cases = json.loads(raw_json)
                print(f"  [Stage2] Fixed! Got {len(test_cases)} test cases")
            except json.JSONDecodeError:
                print("  [Stage2] Still failed after fix")
                test_cases = None
    else:
        print(f"  [Stage2] Tags not found properly. start={tc_start}, end={tc_end}")
        # fallback: 尝试从整个内容中找JSON数组
        array_match = re.search(r'(\[\s*\{[\s\S]*?\}\s*\])', all_content)
        if array_match:
            try:
                test_cases = json.loads(array_match.group(1))
                print(f"  [Stage2] Fallback: found {len(test_cases)} items from content")
            except json.JSONDecodeError:
                pass
            test_cases = None

    return all_content, policy_doc, test_cases
