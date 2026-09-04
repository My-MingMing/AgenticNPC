"""NGRPO estimator registration for a locked veRL installation.

veRL intentionally allows string-named estimators, so this small adapter can
be imported before ``main_ppo``.  It is not imported by inference or tests
unless a training launcher explicitly selects NGRPO.
"""

from __future__ import annotations

from collections import defaultdict


def register_ngrpo() -> None:
    """Register ``ngrpo`` in the active veRL process (idempotent)."""
    import torch
    from verl.trainer.ppo import core_algos

    if "ngrpo" in core_algos.ADV_ESTIMATOR_REGISTRY:
        return

    @core_algos.register_adv_est("ngrpo")
    def compute_ngrpo_outcome_advantage(
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        index,
        epsilon: float = 1e-6,
        config=None,
        **kwargs,
    ):
        """Normalized-GRPO with a reward-unit standard-deviation floor.

        Unlike DAPO, NGRPO keeps per-group standardization and does not filter
        groups or alter response length rewards.  The 0.05 floor prevents a
        single noisy judge result from producing an unbounded policy update.
        """
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
                    means[group_id] = values_tensor.mean()
                    stds[group_id] = values_tensor.std(unbiased=False).clamp_min(0.05)
            for row, group_id in enumerate(index):
                scores[row] = (scores[row] - means[group_id]) / (stds[group_id] + epsilon)
        advantages = scores.unsqueeze(-1) * response_mask
        return advantages, advantages

