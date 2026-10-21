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
   - LOCO runs export their held-out cohort automatically from their `splits_subdir`;
   - `reference_*` and `box_only_*` at `NOISE=5`, `10`, `20` with `NOISE_MODE=loose`, written
     to `inference_output/camera_ready/loose<n>/` (added after the uniform results; see below).
5. **Tables**: `bapmos.results.collate_seeds` for mean ± std over seeds,
   `bapmos.results.paired_stats` for reference vs each ablation/control, and
   `bapmos.results.bandit_preferences` over `reference_*` (and `global_*`).

## Loose-box condition (declared before it was run)

Added after the uniform-perturbation results were known, so it is declared here before any
loose-box export exists, and it is reported whatever it shows.

- **Condition:** every box edge moves outward by an independent U{0..n} px (n = 5, 10, 20),
  clipped to the image: a box drawn larger than the organ. Points are unchanged. Offsets are
  seeded per (slice, n) on their own stream (`<slice>|loose<n>`), identical for both models.
- **Models:** the existing `reference_*` and `box_only_*` checkpoints, seeds 42/43/44; no
  retraining. The uniform-noise exports and every other export are untouched (new folders
  only; the uniform stream keeps its key, and a golden test pins its offsets).
- **Analysis:** the same as for uniform noise: PTV HD95 (primary), all-organ HD95, Dice and
  MSD as mean ± std over seeds; paired Wilcoxon per organ (slice level, Holm-corrected) and
  over all slice-organ pairs, BAP-MOS vs box-only.
- **Caveat known in advance:** training already enlarged every box outward by up to 2 px,
  so this condition is closer to the training distribution than the uniform one.

## Paired robustness for every prompted model (declared before it was run)

The shifted-box exports above (`NOISE_MODE=uniform`) draw all of a slice's offsets from one
stream in organ order, and points draw from it too, so two models with different prompt types
get the same offset distribution but not identical boxes. The loose exports do get identical
boxes for the BAP-MOS family, but the fixed-box SAM/MedSAM baselines had no perturbation hook.

- **Design:** `shift_paired` (edges U{-n..n}) and `loose_paired` (edges outward U{0..n}),
  n = 5, 10, 20, with offsets keyed only by (slice, mask class id, mode, n), so every model
  receives identical boxes; points (BAP-MOS family) use a separate stream (`shift_paired`) or
  stay unchanged (`loose_paired`). Every export writes `prompts_used.json`; the boxes are
  compared across models before any statistics, and a mismatch stops the analysis.
- **Models (existing checkpoints, seeds 42/43/44, test time only):** box-only (same loss),
  BAP-MOS (SAM), SAM (box), MedSAM (box), BAP-MOS (MedSAM). Outputs:
  `inference_output/camera_ready/{pshift,ploose}<n>/`.
- **Primary result:** these paired runs replace the earlier `noise<n>`/`loose<n>` exports as the
  robustness results reported in the paper, for all five models; the earlier exports stay on
  disk, and the paper states that they were replaced and whether their conclusions agree.
- **Analysis:** as before (PTV HD95 primary, all-organ HD95, Dice, MSD; paired Wilcoxon per
  organ with Holm, and over all slice-organ pairs), for BAP-MOS (SAM) vs box-only, BAP-MOS (SAM)
  vs SAM (box), and BAP-MOS (MedSAM) vs MedSAM (box).

## PFUS1: scope and paired robustness (declared before it was run)

- **Scope:** PFUS1 compares the prompt-based methods within each initialisation, SAM (box) vs
  BAP-MOS (SAM) and MedSAM (box) vs BAP-MOS (MedSAM), with U-Net as an unprompted reference.
  nnU-Net was first reported on TRUS only because one PFUS1 validation epoch did not finish
  within a 2 h job; adding it was then attempted (below) and abandoned, so nnU-Net is reported
  on TRUS only.
- **TPE configuration:** the paper's PFUS1 selections from the archived stage-1 studies, SAM
  trial 18 and MedSAM trial 35 (`experiments/bladder/camera_ready/*/selected/paper_tpe/`),
  confirmed by the author. The later outer-loop studies' trials 97/107 (in `main`'s
  `selected/` and in the archived `inner_loop/*/pfus1_*` run configs) are not used. Loss,
  training and common settings are identical to `main`'s PFUS1 production config except the
  train box jitter, which is 2 px (as for every box-prompted camera-ready model) instead of
  the 10 px of `configs/bladder/common.yaml` and the archived PFUS1 runs.
- **Robustness:** the TRUS paired design unchanged (`shift_paired`/`loose_paired`, n = 5, 10,
  20, offsets keyed by slice, class, mode and n) for the four prompted models, seeds 42/43/44,
  on their final checkpoints. Outputs: `inference_output/camera_ready/pf_{pshift,ploose}<n>/`;
  `prompts_used.json` is compared across models before any statistics.
- **Analysis:** bladder (the PFUS1 checkpoint objective) HD95 primary, all-organ HD95, Dice and
  MSD as mean ± std over seeds; paired Wilcoxon at patient level (frames averaged per patient)
  within each initialisation, Holm-corrected. U-Net is constant across perturbations.

## TRUS: what the robustness gain is due to (declared before it was run)

- **Question:** is the robustness of BAP-MOS (SAM) under imperfect boxes due to adaptive
  selection, to training on mixed prompts, or simply to more box augmentation?
- **Mixed prompts without adaptation (test time only):** the existing fixed 2:3:5 schedule and
  shared global selector checkpoints (seeds 42/43/44) get the same paired exports as the five
  models above (`shift_paired`/`loose_paired`, n = 5, 10, 20). If they are as robust as
  BAP-MOS (SAM), the gain is attributed to mixed-prompt training, not to adaptive selection.
- **More box augmentation (new training):** the box-only control (same loss, protocol, TPE
  configuration) retrained with outward train box jitter 0–10 px (`box_only_j10`) and 0–20 px
  (`box_only_j20`) per edge, seeds 42/43/44, with the same jitter function as every other
  box-prompted model; only `prompt_geometry.box_jitter_px` changes. Each gets the exact-prompt
  export and the same paired exports. The outward-only jitter matches the enlarged-box error;
  it is not a shift augmentation, and the paper says so.
- **Analysis:** as in the paired robustness section (PTV HD95 primary, all-organ HD95; paired
  Wilcoxon over slice-organ pairs with Holm), BAP-MOS (SAM) vs each of the four new rows.
- **Other fixed ratios (new training, added after the fixed 2:3:5 robustness result was known):**
  the fixed schedule retrained with Box:Point:Both = 1:1:1 (`fixed_111`), 5:1:1 (`fixed_511`,
  box-heavy) and 1:5:1 (`fixed_151`, point-heavy), seeds 42/43/44, same loss, protocol, TPE
  configuration, 2 px train jitter and test-time rule (highest-Q action) as `fixed_235`. Each gets
  the exact-prompt export and the same paired exports. Questions: how widely fixed ratios differ in
  robustness, where BAP-MOS sits among them, and whether robustness follows the share of point
  and combined prompts in training. Reported whatever the outcome.
  Results are reported whatever their direction, including if box-only with larger jitter is
  as robust as BAP-MOS or better.

## Automatic prompts and LOCO for every prompted model (declared before it was run)

The robustness table reported U-Net-derived prompts and leave-one-cohort-out testing for
box-only and BAP-MOS (SAM) only. Added after those results were known, at the author's
request, so that every prompted column of the table is filled; reported whatever it shows.

- **U-Net-derived prompts (test time only):** SAM (box), MedSAM (box) and BAP-MOS (MedSAM),
  seeds 42/43/44, on their existing pooled checkpoints, with boxes built from the seed-matched
  U-Net export (`PROMPT_MASK_DIR=inference_output/camera_ready/unet/unet_pooled_<seed>`), as
  for box-only and BAP-MOS (SAM). The two fixed-box exports gain `--prompt-mask-dir` (the same
  `load_prompt_mask` as the BAP-MOS export); scoring stays on the GT mask.
- **LOCO (new training):** the same three models retrained with case 1 or case 2 held out
  (`splits_loco/heldout_case<1|2>`), seeds 42/43/44, otherwise identical to their pooled runs:
  SAM (box) is the Table I multi-organ box decoder cell (`table1.slurm SPLITS=...`), MedSAM
  (box) the MedSAM baseline (`medsam_box.slurm SPLITS=...`), BAP-MOS (MedSAM) the
  `loco_case<1|2>_medsam_reference_*` experiments (MedSAM init, paper TPE vector). Exact prompts;
  each exports its held-out case.
- **Analysis:** as for the existing rows: PTV and all-organ HD95, mean over seeds; paired
  Wilcoxon over slice-organ pairs for BAP-MOS (SAM) vs SAM (box) and BAP-MOS (MedSAM) vs
  MedSAM (box), which sets the table's +/- marks.

## PFUS1 nnU-Net (declared before it was run)

nnU-Net is added to the PFUS1 table at the author's request, trained and scored like U-Net on
the same patient-level split (bladder checkpoint objective, 3 replicates, batch 128 via DDP on
4 GPUs, as in the paper-era runs). Two implementation changes make it feasible, neither of
which changes any number:

- **Exact boundary metrics, faster:** nearest-boundary distances use an exact k-d tree instead
  of all-pairs matrices, and each organ's distances are computed once for HD95 and MSD. Outputs
  are bit-identical (unit tests against the original per-point loop, and a byte-identical
  per-slice CSV on 100 PFUS1 validation frames); scoring is ~20x faster.
- **Resumable short jobs:** `checkpoint_latest` is written every epoch on PFUS1
  (`NNUNET_SAVE_EVERY=1`), after the epoch's best score and patience counter are updated, and a
  resumed run restores the counter, so it stops exactly where an uninterrupted run would. (The
  TRUS nnU-Net runs are unaffected: no 40-epoch stretch without improvement occurred in a run
  that kept going.)

**Outcome: not reported.** The first three replicates used nnU-Net's SGD (Nesterov, momentum
0.99) at the shared learning rate 1e-4. With batch 128 an epoch is only ~83 updates, and the
runs stopped early with bladder test Dice 0.0002, 0.0031 and 0.1089. Per-organ test Dice matched
the validation pseudo-Dice, so this was the optimizer setting, not the export. They were rerun
with Adam at flat 1e-4 (`NNUNET_OPTIMIZER=adam`), the optimizer every other PFUS1 model uses,
with the same global batch of 128 on 2-4 GPUs. The Adam reruns were stopped after 13-33 epochs,
before early stopping, because GPUs were not available in time for the camera-ready deadline.
Neither set is reported, and the PFUS1 table has no nnU-Net row. The code changes above stay on
the branch; none of them affects a reported number.

## Caveats to report

- LOCO tests on one clinical volume per fold; it is patient-independent but small.
- Slices within a cohort are correlated, so Wilcoxon p-values are optimistic.
- With GT prompts, an organ absent from the prompt mask is never predicted; with predicted
  prompts, a missed organ stays missed. `paired_stats` reports these as invalid counts.
- `fixed_ring` uses r = 10 px (the paper's r_min): the organ-scaled ring exceeds it on 89% and
  97% of slices for two organs (median 22–23 px) and rarely for the other two.
- A decision block is K loader batches, and BAP-MOS takes one optimizer step per slice. On TRUS
  (batch 1) a block is K slices; on PFUS1 (batch 128) it is 128·K slices, about 1–2 blocks per
  epoch. In five of the six PFUS1 BAP-MOS runs the selected checkpoint is still inside the
  30-block round-robin warm-up (t = 3–9 committed blocks; MedSAM seed 42: t = 43), so those rows
  reflect uniformly mixed prompt training with reward-based test-time action selection. Every
  TRUS run leaves the warm-up well before its selected checkpoint (t = 88–309).
- Probe slices where an organ is present but not predicted have no boundary and are left out
  of that block's reward. A point prompt that cannot be sampled falls back to a box but is
  credited to the selected action.
- PFUS1 distances, and so τ, are in pixels (`pixel_spacing: [1, 1]`).

## Results workbook and draft check

`scripts/camera_ready/results/` turns the exports into the results workbook and checks the
paper draft against it. Only aggregate numbers leave the cluster; `collected.json`, the check
output and the workbooks are git-ignored.

1. On GAIVI, from the repository root: `PYTHONPATH=src python scripts/camera_ready/results/collect.py > collected.json`
   (per-seed means, paired Wilcoxon/Holm tests from `bapmos.results.paired_stats`, and the
   identical-box check across robustness exports). Copy `collected.json` next to the scripts.
2. Locally (needs `openpyxl`), in `scripts/camera_ready/results/`:
   - `python paper_check.py <draft.tex>`: compares the draft's tables, text values and p-value
     claims with `collected.json`;
   - `python make_meta.py`: commit, timestamp and check list for the workbook;
   - `python build.py full.xlsx meta.json` then `python build2.py out.xlsx meta.json full.xlsx`:
     the long-format sheets, then a prostate (TRUS) and a bladder (PFUS1) sheet in front;
   - `python verify.py out.xlsx` and `python verify2.py out.xlsx`: read the workbook back and
     recompute every cell and paired row from `collected.json`.

Every cell is mean ± sample SD (ddof=1) over seeds 42/43/44; a cell with fewer than three seeds
is shown as `pending`, never as a partial average.

The prompt-selection dynamics figure comes from the same reference runs. `train.log` keeps every
block's per-organ rewards but only the first block's arms, so `bandit_dynamics.py` (on GAIVI,
`PYTHONPATH=src python scripts/camera_ready/results/bandit_dynamics.py <run_dir> > dyn.json`)
recovers the arms by replaying `bapmos.method.bandit_policy` on the logged rewards, branching
where the 4-decimal rounding cannot order two UCB scores, and accepts only a path that reproduces
the bandit state saved in `best_checkpoint.pth` and `last_checkpoint.pth`. `fig_dynamics.py
dyn.json fig.pdf` (locally, needs matplotlib) draws the figure; the paper uses seed 42.
