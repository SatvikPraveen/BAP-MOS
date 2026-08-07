# Outer loop — Greedy coordinate search ablation

- Config: `version.yaml` (compose: `bapmos_outer_loop_greedy`)
- Suite: `bapmos_bo_greedy`
- Sampler: catalog (not Optuna TPE)
- Default trials: 100 — baseline + repeated 5-D × 7-probe waves
- State file: `greedy_state.json` under the suite studies root (see `src/bapmos/hpo/paths.py`)
- Export → `../../inner_loop/selected/greedy/pooled.yaml`

```bash
python -m bapmos.hpo.study_runner --hpo-suite bapmos_bo_greedy --help
```

See `docs/SEARCH_METHODS.md`.
