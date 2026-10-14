# Outer loop: TPE with MedSAM (prostate)

A separate study from the SAM search in `../tpe/`. It uses the same search space and per-trial
schedule (at most 100 epochs, patience 15), starting from MedSAM weights.

- Config: `version.yaml` (compose version `bapmos_medsam_pooled_outer_loop`)
- Suite: `bapmos_bo_medsam_pooled`
- Sampler: Optuna TPE after 20 start-up trials (1 baseline + 19 Latin-hypercube samples, seed 42)
- Budget: 100 trials
- Export: `../../inner_loop/medsam/selected/pooled.yaml`

```bash
python -m bapmos.hpo.study_runner init --hpo-suite bapmos_bo_medsam_pooled --dataset pooled
HPO_SEQUENTIAL=1 python -m bapmos.hpo.study_runner run \
  --hpo-suite bapmos_bo_medsam_pooled --dataset pooled --n-trials 100
```

Inner-loop training with the committed export:

```bash
python -m bapmos.method.bap_mos_trainer \
  --version bapmos_medsam_pooled --dataset pooled --experiment pooled_seed42
```

See [`docs/INNER_OUTER_LOOP.md`](../../../../../docs/INNER_OUTER_LOOP.md).
