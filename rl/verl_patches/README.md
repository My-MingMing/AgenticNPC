# NGRPO veRL patch

NGRPO is not an estimator shipped by upstream veRL. The launcher loads
`ngrpo_core_algos.register_ngrpo()` through `scripts/verl_entrypoint.py` before
starting `verl.trainer.main_ppo` (or the async recipe). Set
`RL_VARIANT=ngrpo NGRPO_PATCHED_VERL=1` only in an environment where the
target veRL version exposes `verl.trainer.ppo.core_algos.register_adv_est`.

The estimator computes outcome rewards per prompt group, subtracts the group
mean, and divides by `max(population_std, 0.05)`. It deliberately does not
enable DAPO filtering, overlong shaping, or Clip-Higher, so the comparison is
isolated. Run `python scripts/validate_rl_variant.py ngrpo` as a preflight;
the repository does not claim that a GPU training run has been completed.
