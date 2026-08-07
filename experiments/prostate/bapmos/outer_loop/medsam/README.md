# Outer loop — MedSAM TPE (prostate pooled)

Independent of Meta SAM `../tpe/`. Same 5-D search box and per-trial schedule
(**max_epochs 100 / patience 15**, `loss_mode: kervadec`); init from MedSAM weights.

- Config: `version.yaml` (compose: `bapmos_medsam_pooled_outer_loop`)
- Suite: `bapmos_bo_medsam_pooled`
- Sampler: Optuna TPE
- Default trials: 100
- Export → `../../inner_loop/medsam/selected/pooled.yaml`

```bash
python -m bapmos.hpo.study_runner run \
  --hpo-suite bapmos_bo_medsam_pooled \
  --dataset pooled \
  --n-trials 1
```

Production after export:

```bash
python -m bapmos.method.bap_mos_trainer \
  --version bapmos_medsam_pooled --dataset pooled \
  --experiment pooled_seed42 
```

See `docs/INNER_OUTER_LOOP.md`.
