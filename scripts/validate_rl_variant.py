"""Dependency-free preflight for GRPO, DAPO and NGRPO experiments."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.algorithm_variants import build_variant_overrides, get_variant, validate_variant_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", nargs="?", default=os.getenv("RL_VARIANT", "grpo"))
    parser.add_argument("--max-response-length", type=int, default=4096)
    args = parser.parse_args()
    validate_variant_config(args.variant, ngrpo_patched=os.getenv("NGRPO_PATCHED_VERL") == "1")
    cfg = get_variant(args.variant)
    print(f"variant={cfg.name} reward_manager={cfg.reward_manager}")
    for key, value in build_variant_overrides(cfg.name, max_response_length=args.max_response_length).items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
