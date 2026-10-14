# BAP-MOS inner loop (prostate)

Full training with the selected hyperparameters: boundary-weighted loss, composite reward,
max 300 epochs, patience 40, three seeds (42, 43, 44).

The `selected/` files hold the hyperparameters exported from the paper's outer-loop studies
(`selection_meta.generation: 1`), so you can train directly. Re-running `study_runner export`
overwrites them; restore them with `git checkout -- experiments/`.

| Model | Suite (`--version`) | Selected file | Run folder |
|-------|---------------------|---------------|------------|
| BAP-MOS (SAM) | `bapmos` | `selected/tpe/pooled.yaml` | `runs/prostate/bapmos/inner_loop/tpe/` |
| BAP-MOS (MedSAM) | `bapmos_medsam_pooled` | `medsam/selected/pooled.yaml` | `runs/prostate/bapmos/inner_loop/medsam/` |

```bash
python -m bapmos.method.bap_mos_trainer \
  --version bapmos --dataset pooled --experiment pooled_seed42

python -m bapmos.method.bap_mos_trainer \
  --version bapmos_medsam_pooled --dataset pooled --experiment pooled_seed42
```

Use `pooled_seed43_rep2` and `pooled_seed44_rep3` for the other seeds.

For the search-strategy comparison, train with another export. A bare name (`random`) also
works:

```bash
BAPMOS_SELECTED_SUBDIR=selected/random python -m bapmos.method.bap_mos_trainer \
  --version bapmos --dataset pooled --experiment pooled_seed42
# -> runs/prostate/bapmos/inner_loop/random/pooled_seed42/
```

Always pass `--version` and `--dataset`. `--config …/inner_loop/version.yaml` loads the file as-is
and skips the `selected/` merge. Test export and collation:
[`docs/RUNNING.md`](../../../../docs/RUNNING.md#test-export).
