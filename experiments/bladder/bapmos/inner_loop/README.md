# BAP-MOS inner loop (PFUS1)

Full training with the selected hyperparameters: max 100 epochs, patience 20, three seeds
(42, 43, 44).

| Suite (`--version`) | Selected file | Embeddings | In the paper |
|---------------------|---------------|------------|--------------|
| `bapmos_medsam` | `medsam/selected/pfus1.yaml` | `data/bladder/pfus1/sam_embeddings/medsam_vit_b` | PFUS1 table |
| `bapmos_sam` | `sam/selected/pfus1.yaml` | `data/bladder/pfus1/sam_embeddings/sam_vit_b` | Additional experiment, not reported in the paper |

The `selected/` files hold the hyperparameters exported from the paper's outer-loop studies
(`selection_meta.generation: 1`), so you can train directly. Re-running `study_runner export`
overwrites them; restore them with `git checkout -- experiments/`.

Both suites read cached image embeddings. Build them first
([`docs/PREPROCESS.md`](../../../../docs/PREPROCESS.md#bladder-pfus1)).

```bash
python -m bapmos.method.bap_mos_trainer \
  --version bapmos_medsam --dataset pfus1 --experiment pfus1_seed42
```

Use `pfus1_seed43_rep2` and `pfus1_seed44_rep3` for the other seeds. Checkpoints go to
`runs/bladder/bapmos/inner_loop/{sam,medsam}/<experiment>/`.

Always pass `--version` and `--dataset`; `--config` loads a YAML as-is and skips the `selected/`
merge. Test export and collation: [`docs/RUNNING.md`](../../../../docs/RUNNING.md#inner-loop-and-test-export).
