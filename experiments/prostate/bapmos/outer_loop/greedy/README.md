# Outer loop: greedy coordinate search

- Config: `version.yaml` (compose version `bapmos_outer_loop_greedy`)
- Suite: `bapmos_bo_greedy`
- Sampler: catalog (not Optuna TPE)
- Budget: 100 trials: a baseline trial, then repeated sweeps over the 5 dimensions with 7 probes
  each
- State: `greedy_state.json` next to the study database under `optuna_studies/`
- Export: `../../inner_loop/selected/greedy/pooled.yaml`

`init` enqueues the baseline trial. `run` then finalizes each wave and enqueues the next one.

```bash
python -m bapmos.hpo.study_runner init --hpo-suite bapmos_bo_greedy --dataset pooled
HPO_SEQUENTIAL=1 python -m bapmos.hpo.study_runner run \
  --hpo-suite bapmos_bo_greedy --dataset pooled --n-trials 100
```

To step through the waves by hand, use `greedy-baseline-finalize`, `greedy-wave --wave <0-4>`
and `greedy-finalize --wave <0-4>`. See
[`docs/SEARCH_METHODS.md`](../../../../../docs/SEARCH_METHODS.md#greedy).
