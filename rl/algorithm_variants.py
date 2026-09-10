"""Shared math and launch contracts for the GRPO/DAPO/NGRPO ablation.

DAPO is selected through veRL's DAPO reward manager and trainer options.
NGRPO additionally needs the optional estimator registration in
``rl/verl_patches`` because upstream veRL does not ship that name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class VariantConfig:
    name: str
    adv_estimator: str
    reward_manager: str
    clip_low: float
    clip_high: float
    normalize_by_std: bool
    dynamic_sampling: bool
    overlong_shaping: bool
    kl_in_reward: bool
    std_floor: float = 1e-6
    anchor_reward: float | None = None
    requires_verl_patch: bool = False


VARIANTS = {
    "grpo": VariantConfig("grpo", "grpo", "naive", 0.20, 0.20, True, False, False, True),
    "dapo": VariantConfig("dapo", "grpo", "dapo", 0.20, 0.28, False, True, True, False),
    # NGRPO is this project's Normalized-GRPO ablation: keep GRPO's
    # critic-free group normalization with a std floor, but disable DAPO-only
    # sampling and overlong shaping.
    "ngrpo": VariantConfig(
        "ngrpo", "ngrpo", "naive", 0.20, 0.20, True, False, False, True,
        std_floor=0.05, requires_verl_patch=True,
    ),
    # AiCamp's article uses a different "NGRPO": append a virtual 1.0
    # reward when a real group has zero variance.  Keep a distinct name so it
    # cannot be confused with this project's std-floor estimator.
    "anchor_grpo": VariantConfig(
        "anchor_grpo", "anchor_grpo", "naive", 0.20, 0.20, True,
        False, False, True, std_floor=0.05, anchor_reward=1.0,
        requires_verl_patch=True,
    ),
}


# Each preset changes exactly one GRPO mechanism.  They are intentionally
# separate from VARIANTS: DAPO is a bundle, while these presets answer which
# part of that bundle helps a short-answer, multi-turn NPC workload.
ABLATIONS: dict[str, dict[str, object]] = {
    "baseline": {},
    "clip_high": {
        "actor_rollout_ref.actor.clip_ratio_high": 0.28,
    },
    "token_mean": {
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
    },
    "kl_off": {
        "algorithm.use_kl_in_reward": False,
        "actor_rollout_ref.actor.use_kl_loss": False,
    },
    "dynamic_sampling": {
        "algorithm.filter_groups.enable": True,
        "algorithm.filter_groups.metric": "seq_final_reward",
        "algorithm.filter_groups.max_num_gen_batches": 4,
    },
    "overlong_shaping": {
        # The DAPO reward manager is the veRL component that consumes the
        # buffer settings.  The policy estimator remains stock GRPO.
        "reward_model.reward_manager": "dapo",
        "reward_model.reward_kwargs.overlong_buffer_cfg.enable": True,
        "reward_model.reward_kwargs.overlong_buffer_cfg.len": 512,
        "reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor": 1.0,
    },
}


def get_variant(name: str) -> VariantConfig:
    try:
        return VARIANTS[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"unknown RL variant {name!r}; choose grpo, dapo, ngrpo or anchor_grpo"
        ) from exc


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
    if cfg.anchor_reward is not None and variance**0.5 <= eps:
        augmented = [float(r) for r in rewards] + [cfg.anchor_reward]
        augmented_mean = sum(augmented) / len(augmented)
        augmented_variance = sum(
            (value - augmented_mean) ** 2 for value in augmented
        ) / len(augmented)
        std = max(augmented_variance**0.5, cfg.std_floor, eps)
        return [(float(reward) - augmented_mean) / std for reward in rewards]
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
        "algorithm.adv_estimator": cfg.adv_estimator,
        "algorithm.norm_adv_by_std_in_grpo": cfg.normalize_by_std,
        "algorithm.use_kl_in_reward": cfg.kl_in_reward,
        "actor_rollout_ref.actor.use_kl_loss": cfg.kl_in_reward,
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


def build_ablation_overrides(
    name: str,
    *,
    max_response_length: int = 4096,
) -> dict[str, object]:
    """Return one-factor overrides on top of the stock GRPO baseline.

    ``baseline`` is empty.  The overlong preset also supplies the response
    length consumed by veRL's reward manager.  Presets are kept small so an
    experiment manifest can state exactly what changed.
    """
    key = name.lower()
    try:
        overrides = dict(ABLATIONS[key])
    except KeyError as exc:
        choices = ", ".join(ABLATIONS)
        raise ValueError(f"unknown ablation {name!r}; choose {choices}") from exc
    if key == "overlong_shaping":
        overrides["reward_model.reward_kwargs.max_resp_len"] = max_response_length
    return overrides


def apply_overrides(
    base: Mapping[str, object],
    extra: Mapping[str, object],
) -> dict[str, object]:
    """Merge experiment overrides without mutating either input."""
    merged = dict(base)
    merged.update(extra)
    return merged


def validate_variant_config(name: str, *, ngrpo_patched: bool = False) -> None:
    """Fail fast on unsafe or silently-incorrect experiment combinations."""
    cfg = get_variant(name)
    if cfg.requires_verl_patch and not ngrpo_patched:
        raise RuntimeError(
            f"{cfg.name} requires the project veRL estimator patch; set "
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


def classify_reward_groups(
    groups: Iterable[Sequence[float]],
    *,
    min_reward: float = 0.0,
    max_reward: float = 1.0,
    tolerance: float = 1e-6,
) -> dict[str, int]:
    """Split zero-variance groups by their NPC curriculum meaning.

    A single zero-std ratio conflates mastered prompts, failed prompts and
    equal partial rewards caused by a coarse reward function.
    """
    stats = {"all_pass": 0, "all_fail": 0, "equal_mid": 0, "mixed": 0, "empty": 0}
    for group in groups:
        values = [float(value) for value in group]
        if not values:
            stats["empty"] += 1
        elif max(values) - min(values) > tolerance:
            stats["mixed"] += 1
        elif abs(values[0] - max_reward) <= tolerance:
            stats["all_pass"] += 1
        elif abs(values[0] - min_reward) <= tolerance:
            stats["all_fail"] += 1
        else:
            stats["equal_mid"] += 1
    return stats


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
