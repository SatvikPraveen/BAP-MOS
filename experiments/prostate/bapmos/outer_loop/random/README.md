# Outer loop: random search

- Config: `version.yaml` (compose version `bapmos_outer_loop_random`)
- Suite: `bapmos_bo_random`
- Sampler: Optuna `RandomSampler`, same search space as TPE
- Budget: 100 trials
- Export: `../../inner_loop/selected/random/pooled.yaml`

```bash
python -m bapmos.hpo.study_runner init --hpo-suite bapmos_bo_random --dataset pooled
HPO_SEQUENTIAL=1 python -m bapmos.hpo.study_runner run \
  --hpo-suite bapmos_bo_random --dataset pooled --n-trials 100
```

See [`docs/SEARCH_METHODS.md`](../../../../../docs/SEARCH_METHODS.md).
