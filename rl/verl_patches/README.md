# GRPO estimator patches

These estimators are not shipped by upstream veRL. The launcher loads
`ngrpo_core_algos.register_ngrpo()` through `scripts/verl_entrypoint.py` before
starting `verl.trainer.main_ppo` (or the async recipe). Set
`RL_VARIANT=ngrpo NGRPO_PATCHED_VERL=1` only in an environment where the
target veRL version exposes `verl.trainer.ppo.core_algos.register_adv_est`.

Two explicitly different estimators are registered:

- `ngrpo`: the project's historical std-floor variant. It subtracts the real
  group mean and divides by `max(population_std, 0.05)`. It limits near-zero
  variance amplification but leaves an exactly equal-reward group at zero.
- `anchor_grpo`: the virtual-maximum mechanism described in AiCamp's two GRPO
  articles. Only for an exactly zero-std group, it computes mean/std after
  appending a virtual reward `1.0`; the virtual row is not trained. This can
  create a gradient but cannot identify which real completion is better.

Neither estimator enables DAPO filtering, overlong shaping, or Clip-Higher.
Run `python scripts/validate_rl_variant.py ngrpo` or `anchor_grpo` as a
preflight. The repository does not claim that a GPU training run has completed.
