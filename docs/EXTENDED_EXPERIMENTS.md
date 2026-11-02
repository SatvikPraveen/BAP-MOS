# Extended experiments

Controls, ablations and robustness tests of the BAP-MOS inner loop, run for the camera-ready
version of the paper. They use the same code, data layout and environment as the main
pipeline ([`RUNNING.md`](RUNNING.md)); nothing here changes the README tables or the existing
`selected/` configurations.

## Suites

| Suite (`--version`) | Dataset | Location |
|---------------------|---------|----------|
| `bapmos_camera_ready` | pooled prostate (TRUS) | `experiments/prostate/camera_ready/` |
| `bapmos_camera_ready_pfus1_sam` | PFUS1 | `experiments/bladder/camera_ready/sam/` |
| `bapmos_camera_ready_pfus1_medsam` | PFUS1 | `experiments/bladder/camera_ready/medsam/` |

Each suite starts from the production inner loop with the paper's TPE configuration, stored in
the suite's own `selected/paper_tpe/` (prostate: trial 61, (τ, β, r_min, W, K) =
(26, 0.108, 10, 50, 70)). Every variant changes one setting through its experiment
`overrides` and is trained with seeds 42, 43 and 44 ([`SEEDS.md`](SEEDS.md)).
`reference_*` is the unchanged method and the row every variant is compared with.

| Variant | What changes |
|---------|--------------|
| `box_only` | box prompts only, same loss and protocol (matched control) |
| `fixed_235`, `fixed_111`, `fixed_511`, `fixed_151` | fixed Box:Point:Both schedule instead of the bandit |
| `global` | one shared selector instead of one per organ |
| `dice_reward` | Dice-only reward |
| `fixed_ring` | fixed negative-ring radius (r = r_min) instead of the organ-scaled ring |
| `no_boundary_loss` | CE + Dice without the boundary loss |
| `probe_disjoint` | reward probe disjoint from the checkpoint-selection split |
| `box_only_j10`, `box_only_j20` | box-only with 0–10 / 0–20 px outward training jitter |
| `medsam_reference` | MedSAM initialisation |
| `loco_case<1,2>_*` | leave-one-cohort-out: case 1 or case 2 held out |

## Running

From the repository root, after the data and weights are prepared
([`PREPROCESS.md`](PREPROCESS.md), [`WEIGHTS.md`](WEIGHTS.md)):

```bash
# Leave-one-cohort-out split lists (once)
python -m bapmos.preprocess.prostate.create_loco_splits

# Train one variant and seed
python -m bapmos.method.bap_mos_trainer --version bapmos_camera_ready \
  --dataset pooled --experiment global_seed42 --run-name global_seed42

# PFUS1
python -m bapmos.method.bap_mos_trainer --version bapmos_camera_ready_pfus1_sam \
  --dataset pfus1 --experiment pfus1_seed42 --run-name pfus1_seed42
```

Baselines use their own trainers with the same splits and seeds: `bapmos.external_baselines.unet`,
`bapmos.external_baselines.medsam_init`, `bapmos.external_baselines.nnunet2d` and the
multi-organ box decoder in `bapmos.multiorgan`.

## Test-time prompt conditions

`bapmos.method.run_test_inference` exports a checkpoint on the test split. By default prompts
come from the ground-truth mask without jitter. The options below perturb or replace them;
scoring always uses the ground-truth mask.

| Option | Condition |
|--------|-----------|
| `--prompt-noise-px n --prompt-noise-mode shift_paired` | every box edge shifted by U{−n..n} px |
| `--prompt-noise-px n --prompt-noise-mode loose_paired` | every box edge moved outward by U{0..n} px |
| `--prompt-mask-dir <dir>` | prompts built from another model's predicted masks (e.g. U-Net) |

The `*_paired` modes key the offsets by slice, organ, mode and n, so every model receives
identical boxes; each export writes `prompts_used.json` so this can be checked. The fixed-box
baselines' exporters accept the same options.

## Tables and statistics

- `bapmos.results.collate_seeds`: mean ± std over seeds ([`RESULTS.md`](RESULTS.md)).
- `bapmos.results.paired_stats --a <exports> --b <exports> --unit {slice,patient}`: paired
  Wilcoxon signed-rank tests per organ and over all slice–organ pairs, Holm-corrected.
- `bapmos.results.bandit_preferences`: per-organ prompt preferences (Q and pull counts).
- `scripts/analysis/bandit_dynamics.py` and `scripts/analysis/fig_dynamics.py`: the
  prompt-selection dynamics figure of one run, recovered from `train.log` and verified against
  the bandit state saved in the run's checkpoints.

## Notes

- Leave-one-cohort-out tests on one clinical volume per fold: patient-independent, but small.
- Slices within a cohort are correlated, so slice-level p-values are optimistic.
- PFUS1 distances are in pixel-equivalent units (`pixel_spacing: [1, 1]`).
- Interrupted runs resume with the selection state committed in `last_checkpoint.pth`, so a
  resumed run selects the same checkpoint as an uninterrupted one
  (`tests/test_resume_selection.py`).
