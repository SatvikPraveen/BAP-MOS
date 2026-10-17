# Camera-ready experiments (reviewer response)

Branch `camera-ready-experiments`. Every run below uses this branch's code, the pinned data
snapshot and one environment built from `requirements.txt`. Nothing here changes `main`,
the README tables or `selected/`.

## What each reviewer concern maps to

| Concern | Experiment | Where |
|---------|------------|-------|
| Test-time prompts come from GT masks (R1.2, R2[1], R3 W3/Q2) | Prompt noise ±5/10/20 px; fully automatic prompts from U-Net predictions | `export.slurm` with `NOISE` / `PROMPT_MASK_DIR` |
| Slice-level TRUS split (R1.1, R3 W4/Q3) | Leave-one-cohort-out (case1, case2 held out) | `loco_*` experiments, `unet.slurm SPLITS=...` |
| Per-organ vs global selector (R2[3], R3 W1/Q1) | `global` | camera-ready suite |
| Component ablations (R1.3, R3 W1/Q1) | `dice_reward`, `fixed_ring`, `no_boundary_loss` | camera-ready suite |
| Gain from loss vs bandit (matched controls) | `box_only` (SAM box), `fixed_235` under the BAP-MOS protocol | camera-ready suite |
| Validation reuse (R2[2]) | `probe_disjoint` | camera-ready suite |
| Significance, per-organ metrics (R3 W2/Q5) | Paired Wilcoxon per organ, Holm | `bapmos.results.paired_stats` |
| Per-organ prompt preferences (R3 Q4, R2[3]) | Q/N per organ per seed | `bapmos.results.bandit_preferences` |

The suite is `experiments/prostate/camera_ready/version.yaml` (`--version bapmos_camera_ready`):
the production inner loop with the paper's TPE configuration, one override per variant, seeds
42/43/44. `reference_*` is the unchanged method and is the row every ablation is compared to.

The paper reports TPE trial 61, (τ, α, r_min, W, K) = (26, 0.108, 10, 50, 70), from the archived
study `bapmos_stage1_pooled_ptv_kfold_sclip` (111 completed trials). `main`'s
`experiments/prostate/bapmos/inner_loop/selected/tpe/` holds a later study's trial 52 instead.
The suite therefore carries the paper's vector in its own `selected/paper_tpe/pooled.yaml` and
leaves `main`'s `selected/` unchanged.

## Uniformity

Every row of a table is trained and scored under the same conditions, per dataset:

| Setting | TRUS (pooled) | PFUS1 |
|---------|---------------|-------|
| Split | slice-level 70/15/15 (seed 42), plus LOCO | patient-level 70/15/15 (seed 42) |
| Seeds | 42, 43, 44 (nnU-Net: three unseeded replicates) | same |
| Epochs / patience | 300 / 40 | 100 / 20 |
| Learning rate | 1e-4 | 1e-4 |
| Checkpoint selection | 5-fold validation MSD of the PTV | 5-fold validation MSD of the bladder |
| Box jitter (box-prompted methods) | outward 0–2 px per edge, training only | same |
| Test prompts (promptable methods) | from the GT mask, no jitter, highest-Q action | same |
| Scoring | native grid, one export path, mean ± std over seeds | same |
| Data / environment | pinned snapshot (DATA_SHA256_*.txt), one env per toolkit, GPU logged | same |

What differs between methods is the method itself: BAP-MOS trains with CE + Dice + boundary
loss and Adam; the SAM and MedSAM baselines use CE + Dice and AdamW (weight decay 1e-4); U-Net
uses CE + Dice; nnU-Net uses its own pipeline. The matched controls (`box_only`, `fixed_235`)
run inside the BAP-MOS trainer with the same loss, optimizer, schedule and data path, so each
row of the ablation table differs from BAP-MOS in one factor only.

## Layout on the cluster

```text
${CR_ROOT}/                       default /data/satvikpraveen/camera_ready
  BAP-MOS/                        clone of this branch
    data/prostate/{pooled,case1,case2,simulation}   pinned snapshot (git-ignored)
    models/{sam_base,medsam}/     weights (md5 checked against the official files)
    runs/prostate/camera_ready/<experiment>/
    runs/prostate/camera_ready_baselines/unet/<name>/
    inference_output/camera_ready/...
  env/                            python env with the requirements.txt pins
  DATA_SHA256.txt                 sha256 of every data file (prep.sh)
  logs/                           Slurm logs
  cache/                          pip / torch / matplotlib / tmp
```

## Order

1. **Prep, once** (login node): `bash scripts/camera_ready/prep.sh`. It writes the LOCO
   lists and pins the data. Jobs only verify the manifest; they never rebuild it.
2. **Time one run** before the wave: `sbatch --export=ALL,EXP=reference_seed42
   scripts/camera_ready/train.slurm`, then read start-up and epoch times from its log.
3. **Training wave**: every experiment in the suite (36 runs) and U-Net for seeds 42/43/44 on
   the pooled split and on both LOCO splits (9 runs).
4. **Exports** (`export.slurm`), every seed:
   - each suite run at `NOISE=0`;
   - `reference_*` and `box_only_*` at `NOISE=5`, `10`, `20`;
   - `reference_*` and `box_only_*` with `PROMPT_MASK_DIR=inference_output/camera_ready/unet/
     unet_pooled_<seed>` (the U-Net export root; each slice's `*_pred_ids.png` is found under
     its site), pairing BAP-MOS seed k with U-Net seed k;
   - LOCO runs export their held-out cohort automatically from their `splits_subdir`.
5. **Tables**: `bapmos.results.collate_seeds` for mean ± std over seeds,
   `bapmos.results.paired_stats` for reference vs each ablation/control, and
   `bapmos.results.bandit_preferences` over `reference_*` (and `global_*`).

## Caveats to report

- LOCO tests on one clinical volume per fold; it is patient-independent but small.
- Slices within a cohort are correlated, so Wilcoxon p-values are optimistic.
- With GT prompts, an organ absent from the prompt mask is never predicted; with predicted
  prompts, a missed organ stays missed. `paired_stats` reports these as invalid counts.
- `fixed_ring` uses r = 10 px (the paper's r_min): the organ-scaled ring exceeds it on 89% and
  97% of slices for two organs (median 22–23 px) and rarely for the other two.
