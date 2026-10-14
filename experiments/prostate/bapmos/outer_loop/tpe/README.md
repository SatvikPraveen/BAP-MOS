# Outer loop: TPE (main search)

- Config: `version.yaml` (compose version `bapmos_outer_loop`)
- Suite: `bapmos_bo`
- Sampler: Optuna TPE
- Budget: 100 trials, starting with 20 start-up trials (1 baseline + 19 Latin-hypercube samples,
  seed 42) that `init` enqueues
- Export: `../../inner_loop/selected/tpe/pooled.yaml`

```bash
python -m bapmos.hpo.study_runner init --hpo-suite bapmos_bo --dataset pooled
HPO_SEQUENTIAL=1 python -m bapmos.hpo.study_runner run \
  --hpo-suite bapmos_bo --dataset pooled --n-trials 100
```

`study_runner export --hpo-suite bapmos_bo --dataset pooled` overwrites the committed export
with your study's best trial. See [`docs/SEARCH_METHODS.md`](../../../../../docs/SEARCH_METHODS.md).
