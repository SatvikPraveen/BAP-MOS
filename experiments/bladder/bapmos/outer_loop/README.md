# BAP-MOS outer loop (PFUS1)

TPE search over the five BAP-MOS hyperparameters, scored by fold-balanced validation MSD on the
`Bladder` organ. Each trial trains for at most 15 epochs (patience 4). The bandit warm-up is
shorter than in the inner loop: 5 blocks per arm and 3 pulls before UCB (inner loop: 10 and 5).

| Suite (`--hpo-suite`) | Config | Exports to |
|-----------------------|--------|------------|
| `bapmos_bo_medsam` | `medsam/version.yaml` | `../inner_loop/medsam/selected/pfus1.yaml` |
| `bapmos_bo_sam` | `sam/version.yaml` | `../inner_loop/sam/selected/pfus1.yaml` |

The search runs on the patient subset `splits_hpo_tpe_patient_seed4242_p20`. Build it first:

```bash
python -m bapmos.preprocess.bladder.create_hpo_splits --data_root data/bladder/pfus1
```

Then create the study and run the trials:

```bash
python -m bapmos.hpo.study_runner init --hpo-suite bapmos_bo_medsam --dataset pfus1
HPO_SEQUENTIAL=1 python -m bapmos.hpo.study_runner run \
  --hpo-suite bapmos_bo_medsam --dataset pfus1 --n-trials 100
```

The committed `selected/` files already hold the paper's exported values. Export your own
study only if you want to replace them:

```bash
python -m bapmos.hpo.study_runner export --hpo-suite bapmos_bo_medsam --dataset pfus1
```

Use `bapmos_bo_sam` for the SAM backbone. Details:
[`docs/RUNNING.md`](../../../../docs/RUNNING.md#outer-loop-optional-1),
[`docs/SEARCH_METHODS.md`](../../../../docs/SEARCH_METHODS.md).
