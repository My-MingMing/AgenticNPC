"""
数据格式转换脚本：JSON → Parquet (verl兼容)
将Stage 4输出的train.json转换为verl训练所需的parquet格式
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser(description="Convert training JSON to Parquet for verl")
    parser.add_argument("--input", default="../data/train.json", help="Input JSON file")
    parser.add_argument("--output", default="../data/train.parquet", help="Output parquet file")
    args = parser.parse_args()

    input_path = os.path.join(os.path.dirname(__file__), args.input)
    output_path = os.path.join(os.path.dirname(__file__), args.output)

    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is required. Install with: pip install pandas pyarrow")
        sys.exit(1)

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {input_path}")

    # 转换为verl期望的parquet schema
    records = []
    for i, sample in enumerate(data):
        record = {
            "id": i,
            "data_source": sample.get("data_source", "npc_dialogue"),
            "prompt": json.dumps(sample["prompt"], ensure_ascii=False),
            "ability": sample.get("ability", "agentic"),
            "reward_model": json.dumps(sample.get("reward_model", {"style": "rule", "ground_truth": ""})),
            "extra_info": json.dumps(sample.get("extra_info", {}), ensure_ascii=False),
        }
        records.append(record)

    df = pd.DataFrame(records)

    # 确保所有列都是字符串类型（避免Arrow类型冲突）
    for col in df.columns:
        if col != "id":
            df[col] = df[col].astype(str)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
