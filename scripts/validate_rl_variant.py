"""Dependency-free preflight for GRPO, DAPO and NGRPO experiments."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.algorithm_variants import (
    build_ablation_overrides,
    build_variant_overrides,
    apply_overrides,
    get_variant,
    validate_variant_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", nargs="?", default=os.getenv("RL_VARIANT", "grpo"))
    parser.add_argument(
        "--ablation",
        default=os.getenv("NPC_ABLATION", "baseline"),
        help="one-factor GRPO ablation; use only with variant=grpo",
    )
    parser.add_argument("--max-response-length", type=int, default=4096)
    parser.add_argument(
        "--print-effective-overrides",
        action="store_true",
        help="print only the merged variant + ablation Hydra arguments",
    )
    args = parser.parse_args()
    validate_variant_config(args.variant, ngrpo_patched=os.getenv("NGRPO_PATCHED_VERL") == "1")
    if args.variant.lower() != "grpo" and args.ablation.lower() != "baseline":
        raise SystemExit("NPC_ABLATION presets must use RL_VARIANT=grpo")
    cfg = get_variant(args.variant)
    ablation_overrides = build_ablation_overrides(
        args.ablation, max_response_length=args.max_response_length
    )
    variant_overrides = build_variant_overrides(
        cfg.name, max_response_length=args.max_response_length
    )
    effective_overrides = apply_overrides(variant_overrides, ablation_overrides)
    if args.print_effective_overrides:
        for key, value in effective_overrides.items():
            print(f"{key}={value}")
        return
    print(f"variant={cfg.name} ablation={args.ablation.lower()} reward_manager={cfg.reward_manager}")
    for key, value in effective_overrides.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
