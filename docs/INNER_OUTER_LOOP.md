# Outer loop and inner loop

BAP-MOS is trained in two loops:

| Loop | What it does |
|------|--------------|
| Outer loop | Optuna search over five BAP-MOS hyperparameters. Each trial is a short training run scored by validation MSD. Seed 42 only. |
| Inner loop | Full training with the selected hyperparameters, three seeds (42, 43, 44). |

Commands for both loops are in [`RUNNING.md`](RUNNING.md).

## Method in code

| Component | Implementation | Config keys | Source |
|-----------|----------------|-------------|--------|
| Backbone | SAM or MedSAM ViT-B. Image and prompt encoders frozen; only the mask decoder trains (Adam, lr 1e-4). | `common.sam_checkpoint`, `common.lr` | `src/bapmos/method/sam_model.py`, `src/bapmos/method/bap_mos_trainer.py` |
| Actions | {box, point, both} per organ | `bandit.arms` | `src/bapmos/method/bandit_policy.py` |
| Policy | Per-organ variance-bounded UCB-Tuned with a sliding reward window W. Warm-up of 10 blocks per arm and at least 5 pulls before UCB (bladder outer loop: 5 and 3). | `bandit.window_size`, `bandit.warmup_blocks_per_arm`, `bandit.min_pulls_per_arm` | `src/bapmos/method/bandit_policy.py` |
| Decision block | Each organ's action is held for K training batches. | `bandit.block_size_batches` | `src/bapmos/method/bandit_policy.py` |
| Probe | After each block, a fixed validation probe (20 images, seed 42) scores the block's actions. | `bandit.probe_size`, `bandit.probe_seed` | `src/bapmos/method/bap_mos_trainer.py` |
| Reward | Composite reward bounded in [0, 1] (formula below), τ = clip, λ = 3. | `reward.mode: composite_fixed_clip`, `reward.clip_max_mm`, `reward.hd95_lambda` | `src/bapmos/method/reward.py` |
| Point prompt | 1 positive point inside the organ, 3 negatives from the ring Ω_o = dilate(M_o, r_o) \ M_o with r_o = max(r_min, ⌊α√A_o⌋). If no valid ring exists, the point action falls back to a box. | `prompt_geometry.ring_mode: scale_organ`, `prompt_geometry.alpha`, `prompt_geometry.r_min`, `prompt_geometry.num_pos`, `prompt_geometry.num_neg` | `src/bapmos/method/prompt_geometry.py` |
| Box prompt | Organ bounding box with a small random expansion during training (2 px prostate, 10 px PFUS1). | `prompt_geometry.box_jitter_px` | `src/bapmos/method/prompt_geometry.py` |
| Loss | α(t)·[CE + mean Dice] + (1 − α(t))·boundary loss on signed distance maps; α falls linearly from 1.0 to 0.5. | `training.loss_mode: kervadec`, `training.kervadec.*` | `src/bapmos/losses/kervadec_style.py` |
| Checkpoint and search objective | Fold-balanced (5 folds, seed 42) validation MSD on the primary organ: PTV for prostate, Bladder for PFUS1. | `evaluation.objective_metric: ptv_kfold_msd`, `evaluation.kfold_n`, `evaluation.kfold_seed` | `src/bapmos/metrics/checkpoint_selection.py` |
| Search space | τ ∈ [5, 30] (integer), α ∈ [0.05, 0.15], r_min ∈ [2, 20], W ∈ {20, 25, …, 100}, K ∈ {20, 25, …, 100} | `reward.clip_max_mm`, `prompt_geometry.alpha`, `prompt_geometry.r_min`, `bandit.window_size`, `bandit.block_size_batches` | `src/bapmos/hpo/search_space.py` |
| Search | Optuna TPE, 100 trials. The first 20 are a fixed start-up catalog (1 baseline + 19 Latin-hypercube samples, seed 42) that `study_runner init` enqueues. | suite spec | `src/bapmos/hpo/study_runner.py`, `src/bapmos/hpo/startup_catalog.py` |
| Evaluation grid | Every method is scored on the native reference grid with physical spacing (mm for TRUS, pixel units for PFUS1). | — | `src/bapmos/metrics/evaluation_grid.py` |

Reward for organ o:

```math
R = 1 - \left[0.3\,(1-\mathrm{Dice}) + 0.5\,\frac{\min(\mathrm{MSD},\tau)}{\tau} + 0.2\,\frac{\min(\mathrm{HD95},3\tau)}{3\tau}\right]
```

Trial numbers in a study can exceed 99: failed or pruned trials are purged and re-run, so the
100 completed trials do not always carry numbers 0–99.

## Training schedules

| | Prostate (pooled) | PFUS1 |
|--|--|--|
| Inner loop | max 300 epochs, patience 40 | max 100 epochs, patience 20 |
| Outer-loop trial | max 100 epochs, patience 15 | max 15 epochs, patience 4 |
| Bandit warm-up / min pulls | 10 / 5 (both loops) | inner 10 / 5, outer 5 / 3 |

Shared defaults live in `configs/common/protocol.yaml`, then `configs/<corpus>/common.yaml`,
then the suite's `version.yaml`.

## Loss by method

| Method | Loss |
|--------|------|
| BAP-MOS (SAM or MedSAM) | Boundary-weighted loss above (`training.loss_mode: kervadec`), enforced by the trainer |
| U-Net, MedSAM box baseline, SAM box / point, fixed schedules, legacy policies | CE + Dice |
| nnU-Net | nnU-Net's own Dice + CE |

The policy is enforced in `src/bapmos/losses/loss_policy.py`.

## Selected hyperparameters

The committed `experiments/**/inner_loop/**/selected/*.yaml` files hold the hyperparameters
exported from the paper's outer-loop studies (`selection_meta.generation: 1`). The inner loop
runs directly with them.

Running `study_runner export` writes your own study's best trial over the matching file. This is
optional. Restore the paper's values with `git checkout -- experiments/`.

| Suite (`--version`) | Dataset | `selected/` file merged |
|---------------------|---------|-------------------------|
| `bapmos` | `pooled` | `experiments/prostate/bapmos/inner_loop/selected/tpe/pooled.yaml` |
| `bapmos_medsam_pooled` | `pooled` | `experiments/prostate/bapmos/inner_loop/medsam/selected/pooled.yaml` |
| `bapmos_sam` | `pfus1` | `experiments/bladder/bapmos/inner_loop/sam/selected/pfus1.yaml` |
| `bapmos_medsam` | `pfus1` | `experiments/bladder/bapmos/inner_loop/medsam/selected/pfus1.yaml` |

For the prostate search-strategy comparison, set `BAPMOS_SELECTED_SUBDIR=selected/<method>`
(`random`, `greedy` or `heuristic`). Runs then go to
`runs/prostate/bapmos/inner_loop/<method>/`.

## How configuration is composed

| CLI | Behaviour |
|-----|-----------|
| `--version <suite> --dataset <name>` | Merges `configs/<corpus>/common.yaml`, the suite's `version.yaml`, and the `selected/` file. Use this for reproduction. |
| `--config <yaml>` | Loads the YAML as-is. Does not merge the site defaults or `selected/`. |

## Suites

| Corpus | Outer-loop suite (`--hpo-suite`) | Outer config | Inner suite (`--version`) |
|--------|----------------------------------|--------------|---------------------------|
| Prostate, SAM | `bapmos_bo` (TPE); `bapmos_bo_random`, `bapmos_bo_greedy`, `bapmos_bo_heuristic` | `experiments/prostate/bapmos/outer_loop/{tpe,random,greedy,heuristic}/version.yaml` | `bapmos` |
| Prostate, MedSAM | `bapmos_bo_medsam_pooled` | `experiments/prostate/bapmos/outer_loop/medsam/version.yaml` | `bapmos_medsam_pooled` |
| PFUS1, SAM | `bapmos_bo_sam` | `experiments/bladder/bapmos/outer_loop/sam/version.yaml` | `bapmos_sam` |
| PFUS1, MedSAM | `bapmos_bo_medsam` | `experiments/bladder/bapmos/outer_loop/medsam/version.yaml` | `bapmos_medsam` |

The PFUS1 outer loop runs on `splits_hpo_tpe_patient_seed4242_p20`, built by
`bapmos.preprocess.bladder.create_hpo_splits` ([`PREPROCESS.md`](PREPROCESS.md#bladder-pfus1)).

## Checkpoints and test metrics

- Outer loop: after a trial's score is recorded, its `*.pth` files are deleted. A failed trial
  keeps `last_checkpoint.pth` so it can resume.
- Inner loop: checkpoints are kept (`evaluation.cleanup_checkpoints_after_train: false`).
- Inner-loop configs set `evaluation.run_test_after_train: false`. Test metrics come from the
  separate `run_test_inference` export ([`RUNNING.md`](RUNNING.md#test-export)).

## Earlier protocol

The fixed-schedule and policy experiments under `experiments/*/legacy/` use an earlier protocol:
negated validation MSD as reward, a fixed 20 px negative-point ring, and CE + Dice loss. They run
on the trainers in `src/bapmos/legacy/`. See
[`experiments/prostate/legacy/README.md`](../experiments/prostate/legacy/README.md).
