# Outer-loop search methods (prostate)

Hyperparameter search for BAP-MOS (clip τ, organ-scale ring α / r_min, bandit window / block).
Objective: **validation PTV k-fold MSD** (same as checkpointing).

Each search method has its own trial config under `outer_loop/<method>/version.yaml`.

| Method | Suite id (`--hpo-suite`) | Compose version | Config |
|--------|--------------------------|-----------------|--------|
| **TPE** (Meta SAM, main) | `bapmos_bo` | `bapmos_outer_loop` | `outer_loop/tpe/version.yaml` |
| **Random** (Meta SAM) | `bapmos_bo_random` | `bapmos_outer_loop_random` | `outer_loop/random/version.yaml` |
| **Greedy** (Meta SAM) | `bapmos_bo_greedy` | `bapmos_outer_loop_greedy` | `outer_loop/greedy/version.yaml` |
| **Heuristic** (Meta SAM) | `bapmos_bo_heuristic` | `bapmos_outer_loop_heuristic` | `outer_loop/heuristic/version.yaml` |
| **TPE** (MedSAM) | `bapmos_bo_medsam_pooled` | `bapmos_medsam_pooled_outer_loop` | `outer_loop/medsam/version.yaml` |

Per-trial train schedule: **max_epochs 100 / patience 15**, `loss_mode: kervadec` (same loss as production).
Optuna trial count: **100 trials** (see `docs/SEARCH_METHODS.md`).

MedSAM search is **independent** of Meta SAM (`tpe/`): same protocol/budget, MedSAM weights, own Optuna study and selected export.

## Layout

```text
outer_loop/
  tpe/version.yaml          # Meta SAM
  random|greedy|heuristic/
  medsam/version.yaml       # MedSAM (independent TPE)
inner_loop/
  selected/{tpe,random,greedy,heuristic}/pooled.yaml
  version.yaml              # production Meta SAM
  medsam/
    version.yaml            # production MedSAM
    selected/pooled.yaml    # from bapmos_bo_medsam_pooled export
```

### Meta SAM production

```bash
python -m bapmos.method.bap_mos_trainer \
  --version bapmos --dataset pooled --experiment pooled_seed42 \
  
```

### MedSAM production

```bash
python -m bapmos.hpo.study_runner run \
  --hpo-suite bapmos_bo_medsam_pooled --dataset pooled --n-trials 1
# after export:
python -m bapmos.method.bap_mos_trainer \
  --version bapmos_medsam_pooled --dataset pooled \
  --experiment pooled_seed42 
```

Details: `docs/SEARCH_METHODS.md`, `docs/INNER_OUTER_LOOP.md`
