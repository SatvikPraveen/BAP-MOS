# Outer loop — Random search ablation

- Config: `version.yaml` (compose: `bapmos_outer_loop_random`)
- Suite: `bapmos_bo_random`
- Sampler: Optuna RandomSampler
- Default trials: 100
- Same 5-D search box as TPE (`searched_clip_scale_organ`)
- Export → `../../inner_loop/selected/random/pooled.yaml`

```bash
python -m bapmos.hpo.study_runner run \
  --hpo-suite bapmos_bo_random \
  --dataset pooled \
  --n-trials 1
```

See `docs/SEARCH_METHODS.md`.
