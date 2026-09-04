"""Shared math and launch contracts for the GRPO/DAPO/NGRPO ablation.

DAPO is selected through veRL's DAPO reward manager and trainer options.
NGRPO additionally needs the optional estimator registration in
``rl/verl_patches`` because upstream veRL does not ship that name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class VariantConfig:
    name: str
    reward_manager: str
    clip_low: float
    clip_high: float
    normalize_by_std: bool
    dynamic_sampling: bool
    overlong_shaping: bool
    kl_in_reward: bool
    std_floor: float = 1e-6
    requires_verl_patch: bool = False


VARIANTS = {
    "grpo": VariantConfig("grpo", "naive", 0.20, 0.20, True, False, False, True),
    "dapo": VariantConfig("dapo", "dapo", 0.20, 0.28, False, True, True, False),
    # NGRPO is this project's Normalized-GRPO ablation: keep GRPO's
    # critic-free group normalization with a std floor, but disable DAPO-only
    # sampling and overlong shaping.
    "ngrpo": VariantConfig(
        "ngrpo", "naive", 0.20, 0.20, True, False, False, True,
        std_floor=0.05, requires_verl_patch=True,
    ),
}


def get_variant(name: str) -> VariantConfig:
    try:
        return VARIANTS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown RL variant {name!r}; choose grpo, dapo or ngrpo") from exc


def group_advantages(
    rewards: Sequence[float],
    *,
    variant: str = "grpo",
    eps: float = 1e-6,
) -> List[float]:
    """Compute critic-free group advantages for one prompt's responses.

    GRPO uses mean/std normalization.  DAPO's token-level loss consumes the
    un-divided centered reward, while NGRPO keeps normalization but floors the
    standard deviation to avoid exploding a near-constant group.  All outputs
    are finite and preserve response ordering.
    """
    cfg = get_variant(variant)
    if not rewards:
        return []
    mean = sum(float(r) for r in rewards) / len(rewards)
    centered = [float(r) - mean for r in rewards]
    if not cfg.normalize_by_std:
        return centered
    variance = sum(value * value for value in centered) / len(centered)
    # NGRPO uses a configurable floor in reward units.  A floor is important
    # for NPC tasks because a mostly-deterministic group otherwise produces
    # very large ratios from a single judge/tool failure.
    floor = cfg.std_floor if cfg.name == "ngrpo" else eps
    std = max(variance**0.5, floor, eps)
    return [value / std for value in centered]


def build_variant_overrides(name: str, *, max_response_length: int = 4096) -> dict[str, object]:
    """Return the Hydra values shared by all launchers.

    This is deliberately data-only so a launcher can print/inspect the exact
    experiment contract before importing veRL.  It does not claim that a
    target veRL release consumes every optional DAPO field.
    """
    cfg = get_variant(name)
    overrides: dict[str, object] = {
        "algorithm.adv_estimator": "ngrpo" if cfg.requires_verl_patch else "grpo",
        "algorithm.norm_adv_by_std_in_grpo": cfg.normalize_by_std,
        "algorithm.use_kl_in_reward": cfg.kl_in_reward,
        "actor_rollout_ref.actor.clip_ratio_low": cfg.clip_low,
        "actor_rollout_ref.actor.clip_ratio_high": cfg.clip_high,
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean" if cfg.name == "dapo" else "sample-mean",
        "reward_model.reward_manager": cfg.reward_manager,
    }
    if cfg.name == "dapo":
        overrides.update({
            "algorithm.filter_groups.enable": True,
            "algorithm.filter_groups.metric": "seq_final_reward",
            "algorithm.filter_groups.max_num_gen_batches": 4,
            "reward_model.reward_kwargs.max_resp_len": max_response_length,
            "reward_model.reward_kwargs.overlong_buffer_cfg.enable": True,
            "reward_model.reward_kwargs.overlong_buffer_cfg.len": 512,
            "reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor": 1.0,
        })
    return overrides


def validate_variant_config(name: str, *, ngrpo_patched: bool = False) -> None:
    """Fail fast on unsafe or silently-incorrect experiment combinations."""
    cfg = get_variant(name)
    if cfg.requires_verl_patch and not ngrpo_patched:
        raise RuntimeError(
            "NGRPO requires the project veRL estimator patch; set "
            "NGRPO_PATCHED_VERL=1 only after installing that patch."
        )


def filter_degenerate_groups(
    groups: Iterable[Sequence[float]],
    *,
    min_std: float = 1e-6,
) -> Tuple[List[Sequence[float]], int]:
    """DAPO Dynamic Sampling primitive: discard all-equal reward groups."""
    kept: List[Sequence[float]] = []
    discarded = 0
    for group in groups:
        values = [float(value) for value in group]
        if not values:
            discarded += 1
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        if variance**0.5 <= min_std:
            discarded += 1
        else:
            kept.append(group)
    return kept, discarded


def overlong_reward_shaping(
    reward: float,
    response_tokens: int,
    *,
    soft_limit: int,
    hard_limit: int,
    penalty_floor: float = -1.0,
) -> float:
    """Apply DAPO-style soft overlong penalty without hiding hard truncation."""
    if hard_limit <= soft_limit:
        raise ValueError("hard_limit must be greater than soft_limit")
    if response_tokens <= soft_limit:
        return float(reward)
    if response_tokens >= hard_limit:
        return float(penalty_floor)
    ratio = (response_tokens - soft_limit) / (hard_limit - soft_limit)
    return float(reward) * (1.0 - ratio) + float(penalty_floor) * ratio


def token_level_weights(lengths: Sequence[int]) -> List[float]:
    """Return per-response weights for token-level (DAPO) aggregation."""
    total = sum(max(int(length), 0) for length in lengths)
    if total <= 0:
        return [0.0 for _ in lengths]
    return [max(int(length), 0) / total for length in lengths]


def resample_groups(
    sampler: Callable[[], Sequence[float]],
    target_groups: int,
    *,
    max_attempts: int = 100,
    min_std: float = 1e-6,
) -> Tuple[List[Sequence[float]], int]:
    """Repeatedly sample until DAPO has enough non-degenerate groups."""
    if target_groups < 0:
        raise ValueError("target_groups must be non-negative")
    accepted: List[Sequence[float]] = []
    attempts = 0
    while len(accepted) < target_groups and attempts < max_attempts:
        attempts += 1
        kept, _ = filter_degenerate_groups([sampler()], min_std=min_std)
        accepted.extend(kept)
    return accepted[:target_groups], attempts
