# `bapmos.legacy.ablations`

Small **constants and documentation** for reviewer-driven historical sweeps.

- Training code stays in `bapmos.legacy.optimization` and baseline modules.
- Do **not** fork trainers here; wire grids via experiment YAML under
  `experiments/*/legacy/` (see `docs/EXPERIMENT_LADDER.md` historical rungs).
- Optional YAML copies under `configs/ablations/` are **not** shipped in this
  tree; copy from a baseline config when needed.
- `config_presets.py` lists suggested grids (UCB `c`, clip τ, probe size, block
  size) — wire them into those YAML copies.
- **Probe/block grids:** use `INTERNAL_*_GRID` for simulation / Case 1/2 (small
  cohorts); use `PFUS1_*_GRID` for full PFUS1 sweeps. `PROBE_SIZE_GRID` /
  `BLOCK_SIZE_GRID` alias the PFUS1-scale defaults.
