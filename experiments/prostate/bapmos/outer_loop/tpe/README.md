# Outer loop — TPE (default / main paper search)

- Config: `version.yaml` (compose: `bapmos_outer_loop`)
- Suite: `bapmos_bo`
- Sampler: Optuna TPE (`sampler: tpe`)
- Default trials: 100 (startup catalog ~20)
- Optuna study / trial paths: configured in `src/bapmos/hpo/paths.py`
- Export → `../../inner_loop/selected/tpe/pooled.yaml`

```bash
python -m bapmos.hpo.study_runner run \
  --hpo-suite bapmos_bo \
  --dataset pooled \
  --n-trials 1
```

See `docs/SEARCH_METHODS.md`.
