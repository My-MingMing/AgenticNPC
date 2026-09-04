"""Preload project veRL extensions, then execute a veRL module.

Usage: ``python verl_entrypoint.py verl.trainer.main_ppo <hydra overrides>``.
The wrapper keeps stock GRPO/DAPO untouched and is required only for NGRPO.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: verl_entrypoint.py <module> [args ...]")
    module = sys.argv[1]
    if module not in {"verl.trainer.main_ppo", "recipe.one_step_off_policy.main_ppo"}:
        raise SystemExit(f"unsupported veRL entrypoint: {module}")
    if __import__("os").environ.get("RL_VARIANT", "grpo").lower() == "ngrpo":
        from rl.verl_patches.ngrpo_core_algos import register_ngrpo

        register_ngrpo()
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
