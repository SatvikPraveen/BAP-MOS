# Bladder (PFUS1) experiment ladder

Same MSD objective and three-seed protocol as prostate, with
`data_root: data/bladder/pfus1`, `checkpoint_objective.objective_organ: Bladder`,
and defaults from `configs/bladder/common.yaml` (patience **20** / **100** epochs).

## Layout

- `bapmos/outer_loop/{sam,medsam}/` — HPO suites `bapmos_bo_sam` / `bapmos_bo_medsam`
- `bapmos/inner_loop/{sam,medsam}/` — production (selected exports + full train)
- `baselines/{unet,nnunet,sam_box,medsam}/` — main-paper baselines
- `legacy/box/`, `legacy/point/` — historical prompt baselines
- `legacy/policies/` — **not implemented** here (see that README; policies live under prostate)

No prostate-style `tpe/random/greedy/heuristic` search-method family on bladder.
Details: `docs/EXPERIMENT_LADDER.md`, `docs/INNER_OUTER_LOOP.md`.

Legacy prompt protocol: see `legacy/README.md` and `experiments/prostate/legacy/README.md`.
