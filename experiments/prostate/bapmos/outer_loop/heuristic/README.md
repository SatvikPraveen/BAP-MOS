# Outer loop: heuristic catalog

- Config: `version.yaml` (compose version `bapmos_outer_loop_heuristic`)
- Suite: `bapmos_bo_heuristic`
- Sampler: fixed catalog in tiers, deduplicated and capped at 100 trials (no adaptive sampler)
- Same search space as TPE
- Export: `../../inner_loop/selected/heuristic/pooled.yaml`

`init` enqueues the whole catalog; `run` trains the queued trials.

```bash
python -m bapmos.hpo.study_runner init --hpo-suite bapmos_bo_heuristic --dataset pooled
HPO_SEQUENTIAL=1 python -m bapmos.hpo.study_runner run \
  --hpo-suite bapmos_bo_heuristic --dataset pooled --n-trials 100
```

Catalog tiers: [`docs/SEARCH_METHODS.md`](../../../../../docs/SEARCH_METHODS.md#heuristic).
