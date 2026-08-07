# Outer loop — Heuristic catalog search ablation

- Config: `version.yaml` (compose: `bapmos_outer_loop_heuristic`)
- Suite: `bapmos_bo_heuristic`
- Sampler: catalog (fixed multi-tier candidate list; not Optuna TPE)
- Default trials: 100 (deduped catalog, capped)
- Same 5-D search box as TPE (`searched_clip_scale_organ`)
- Export → `../../inner_loop/selected/heuristic/pooled.yaml`

```bash
python -m bapmos.hpo.study_runner run \
  --hpo-suite bapmos_bo_heuristic \
  --dataset pooled \
  --n-trials 1
```

Catalog tiers: see `docs/SEARCH_METHODS.md`.
