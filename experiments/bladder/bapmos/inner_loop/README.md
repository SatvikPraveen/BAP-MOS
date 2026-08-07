# Inner loop — production (bladder / PFUS1)

Full training with **selected** outer-loop hyperparameters.
Schedule: **max_epochs 100 / patience 20**. Three seeds: 42 / 43 / 44.

## Required CLI (compose)

```bash
# Meta SAM
python -m bapmos.method.bap_mos_trainer \
  --version bapmos_sam --dataset pfus1 --experiment pfus1_seed42

# MedSAM
python -m bapmos.method.bap_mos_trainer \
  --version bapmos_medsam --dataset pfus1 --experiment pfus1_seed42
```

`--config` is advanced only (skips `selected/` merge). See `docs/INNER_OUTER_LOOP.md`.

Start inner loop **only after** outer-loop export overwrites `selected/` placeholders.
