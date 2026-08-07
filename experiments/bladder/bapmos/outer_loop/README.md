# BAP-MOS outer loop on PFUS1 — validation Bladder MSD (search)

Objective: fold-balanced validation **Bladder** MSD (`ptv_kfold_msd`, `objective_organ: Bladder`).
Per-trial schedule: **max_epochs 15 / patience 4**, `loss_mode: kervadec`.

Suite and search configuration for PFUS1 bladder are defined under `src/bapmos/hpo/`
and `experiments/bladder/bapmos/`. See `docs/SEARCH_METHODS.md` and `docs/EXPERIMENT_LADDER.md`.

```text
outer_loop/
  sam/version.yaml      # bapmos_bo_sam / bapmos_sam_outer_loop
  medsam/version.yaml   # bapmos_bo_medsam / bapmos_medsam_outer_loop
inner_loop/
  sam/version.yaml + selected/pfus1.yaml
  medsam/version.yaml + selected/pfus1.yaml
```

Prostate search ablations (`tpe` / `random` / `greedy` / `heuristic`) live under
`experiments/prostate/bapmos/outer_loop/`. Bladder currently uses the **main BAP-MOS
protocol only** (no full search-method ablation family in this public tree).

```bash
python -m bapmos.hpo.study_runner --help
# Suites: bapmos_bo_sam / bapmos_bo_medsam (see src/bapmos/hpo/paths.py)
```
