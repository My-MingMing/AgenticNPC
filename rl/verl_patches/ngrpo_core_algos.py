"""NGRPO estimator registration for a locked veRL installation.

veRL intentionally allows string-named estimators, so this small adapter can
be imported before ``main_ppo``.  It is not imported by inference or tests
unless a training launcher explicitly selects NGRPO.
"""

from __future__ import annotations

from collections import defaultdict


def register_ngrpo() -> None:
    """Register the std-floor and zero-variance-anchor estimators."""
    import torch
    from verl.trainer.ppo import core_algos

    def _estimate(
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        index,
        *,
        zero_std_anchor: float | None,
        epsilon: float = 1e-6,
    ):
        scores = token_level_rewards.sum(dim=-1)
        grouped = defaultdict(list)
        for row, group_id in enumerate(index):
            grouped[group_id].append(scores[row])
        means, stds = {}, {}
        with torch.no_grad():
            for group_id, values in grouped.items():
                if len(values) <= 1:
                    means[group_id] = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
                    stds[group_id] = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
                else:
                    values_tensor = torch.stack(values)
                    real_std = values_tensor.std(unbiased=False)
                    if zero_std_anchor is not None and real_std <= epsilon:
                        anchor = torch.tensor(
                            [zero_std_anchor], device=scores.device, dtype=scores.dtype
                        )
                        stats_values = torch.cat((values_tensor, anchor))
                    else:
                        stats_values = values_tensor
                    means[group_id] = stats_values.mean()
                    stds[group_id] = stats_values.std(unbiased=False).clamp_min(0.05)
            for row, group_id in enumerate(index):
                scores[row] = (scores[row] - means[group_id]) / (stds[group_id] + epsilon)
        advantages = scores.unsqueeze(-1) * response_mask
        return advantages, advantages

    if "ngrpo" not in core_algos.ADV_ESTIMATOR_REGISTRY:
        @core_algos.register_adv_est("ngrpo")
        def compute_ngrpo_outcome_advantage(
            token_level_rewards, response_mask, index, epsilon=1e-6, config=None, **kwargs
        ):
            """Project NGRPO: ordinary group normalization with std floor."""
            return _estimate(
                token_level_rewards, response_mask, index,
                zero_std_anchor=None, epsilon=epsilon,
            )

    if "anchor_grpo" not in core_algos.ADV_ESTIMATOR_REGISTRY:
        @core_algos.register_adv_est("anchor_grpo")
        def compute_anchor_grpo_outcome_advantage(
            token_level_rewards, response_mask, index, epsilon=1e-6, config=None, **kwargs
        ):
            """AiCamp-style ablation: add a virtual 1.0 only to zero-std stats."""
            return _estimate(
                token_level_rewards, response_mask, index,
                zero_std_anchor=1.0, epsilon=epsilon,
            )
