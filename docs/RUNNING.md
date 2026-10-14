# Running experiments

This page is the command reference for the repository. Every step is a `python -m …` command
run from the repository root. Wrap any of them in your own batch system if you use one.

Related pages:

- Method, protocol and the two loops: [`INNER_OUTER_LOOP.md`](INNER_OUTER_LOOP.md)
- Data preparation: [`PREPROCESS.md`](PREPROCESS.md)
- Seeds and run names: [`SEEDS.md`](SEEDS.md)
- Result tables: [`RESULTS.md`](RESULTS.md)
- Where outputs are written: [`RUNTIME_LAYOUT.md`](RUNTIME_LAYOUT.md)

---

## 1. Setup

```bash
cd BAP-MOS
pip install -r requirements.txt        # core: training, preprocessing, inference, W&B
pip install -r requirements-hpo.txt    # Optuna (pinned to 4.9.0, the version used for the paper)
pip install -r requirements_dev.txt    # pytest and notebook extras (optional)
# CPU only: pip install -r requirements_cpu.txt
# nnU-Net baseline: separate environment, pip install -r requirements_nnunet.txt

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
```

There is no package install. Modules run with `src/` on `PYTHONPATH`. `./scripts/bapmos.sh`
sets `PYTHONPATH` itself and wraps the preprocessing, test and result-collation steps:

```bash
./scripts/bapmos.sh preprocess-prostate --dry-run
./scripts/bapmos.sh preprocess-bladder --dry-run
./scripts/bapmos.sh test
```

Weights go under `models/` ([`WEIGHTS.md`](WEIGHTS.md)). Data go under `data/prostate/pooled/`
and `data/bladder/pfus1/` ([`PREPROCESS.md`](PREPROCESS.md)).

W&B logging is on by default. Pass `--no-wandb` to the BAP-MOS trainer and to `study_runner`,
or run `wandb offline` / `wandb disabled` for the other trainers.

| File | Contents |
|------|----------|
| `requirements.txt` | Core training, preprocessing, inference and W&B |
| `requirements_cpu.txt` | Core stack with the CPU PyTorch index |
| `requirements_dev.txt` | pytest, Jupyter, Plotly, PyVista |
| `requirements-hpo.txt` | Optuna for the outer loop |
| `requirements_nnunet.txt` | nnU-Net v2 (separate environment) |

---

## 2. BAP-MOS on prostate

### Outer loop (optional)

The selected hyperparameters from the paper's outer-loop studies are committed under
`experiments/prostate/bapmos/inner_loop/selected/`. You can skip this step and go straight to
the inner loop.

To repeat a search, create the study with `init`, then run trials with `run`:

```bash
python -m bapmos.hpo.study_runner init --hpo-suite bapmos_bo --dataset pooled
HPO_SEQUENTIAL=1 python -m bapmos.hpo.study_runner run \
  --hpo-suite bapmos_bo --dataset pooled --n-trials 100
```

- `init` creates the Optuna database under `optuna_studies/` and, for TPE suites, enqueues the
  20 start-up trials (1 baseline + 19 Latin-hypercube samples, seed 42).
- `HPO_SEQUENTIAL=1` runs trials one after another on one machine.
- TPE and random studies stop early once the objective saturates. Set
  `HPO_STOP_ON_SATURATION=0` to run the full trial budget.
- Search trials use seed 42 only.

Other suites follow the same `init` then `run` pattern:

| Suite (`--hpo-suite`) | Search | Writes `selected/` file |
|-----------------------|--------|-------------------------|
| `bapmos_bo` | TPE, SAM (main) | `inner_loop/selected/tpe/pooled.yaml` |
| `bapmos_bo_random` | Random | `inner_loop/selected/random/pooled.yaml` |
| `bapmos_bo_heuristic` | Fixed catalog | `inner_loop/selected/heuristic/pooled.yaml` |
| `bapmos_bo_greedy` | Greedy coordinate search | `inner_loop/selected/greedy/pooled.yaml` |
| `bapmos_bo_medsam_pooled` | TPE, MedSAM | `inner_loop/medsam/selected/pooled.yaml` |

For greedy, `init` enqueues the baseline trial and `run` finalizes each wave and enqueues the
next one automatically. See [`SEARCH_METHODS.md`](SEARCH_METHODS.md).

### Export selected hyperparameters (optional)

```bash
python -m bapmos.hpo.study_runner export --hpo-suite bapmos_bo --dataset pooled
```

This overwrites the committed `selected/` file for that suite with the best trial of your study.
Restore the paper's values with `git checkout -- experiments/`.

### Inner loop (three seeds)

```bash
for exp in pooled_seed42 pooled_seed43_rep2 pooled_seed44_rep3; do
  python -m bapmos.method.bap_mos_trainer \
    --version bapmos --dataset pooled --experiment "${exp}"
done
```

The experiment entry sets the training seed (42, 43, 44) and the run folder name.
Checkpoints go to `runs/prostate/bapmos/inner_loop/tpe/<experiment>/best_checkpoint.pth`.

Other variants:

```bash
# BAP-MOS (MedSAM) -> runs/prostate/bapmos/inner_loop/medsam/<experiment>/
python -m bapmos.method.bap_mos_trainer \
  --version bapmos_medsam_pooled --dataset pooled --experiment pooled_seed42

# Search-strategy comparison: train with the random / greedy / heuristic export
# -> runs/prostate/bapmos/inner_loop/random/<experiment>/
BAPMOS_SELECTED_SUBDIR=selected/random python -m bapmos.method.bap_mos_trainer \
  --version bapmos --dataset pooled --experiment pooled_seed42
```

Always pass `--version` and `--dataset`. `--config` loads a YAML as-is and does not merge the
`selected/` file ([`INNER_OUTER_LOOP.md`](INNER_OUTER_LOOP.md#how-configuration-is-composed)).

### Test export

Test metrics come from a separate export step. The export refuses non-seed-42 runs unless you
pass `--force-inference-output`, so pass it for seeds 43 and 44:

```bash
CKPT_ROOT=runs/prostate/bapmos/inner_loop/tpe
OUT_ROOT=inference_output/prostate/pooled/bapmos

python -m bapmos.method.run_test_inference \
  --checkpoint "${CKPT_ROOT}/pooled_seed42/best_checkpoint.pth" \
  --output_dir "${OUT_ROOT}/pooled_seed42"

for exp in pooled_seed43_rep2 pooled_seed44_rep3; do
  python -m bapmos.method.run_test_inference \
    --checkpoint "${CKPT_ROOT}/${exp}/best_checkpoint.pth" \
    --output_dir "${OUT_ROOT}/${exp}" \
    --force-inference-output
done
```

Each export has one folder per test site (`simulation/`, `case1/`, `case2/`). Metrics cover the
full test split; prediction panels cover the first 10 slices. Use the same pattern for the other
variants, with output folders `bapmos_medsam/`, `bapmos_random/`, `bapmos_greedy/` and
`bapmos_heuristic/` under `inference_output/prostate/pooled/`.

### Result tables

```bash
bash scripts/populate_results_prostate.sh
```

The script ingests every export it finds under `inference_output/prostate/pooled/`, skips
missing runs with a warning, and builds the mean ± std tables. Manual `collate_seeds` commands
are in [`RESULTS.md`](RESULTS.md#manual-collation).

---

## 3. Baselines on prostate

Three seeds per baseline (42, 43, 44). The commands below show seed 42; change `--seed` and the
run name (`…_seed43_rep2`, `…_seed44_rep3`) for the other seeds, and add
`--force-inference-output` to their test export.

### U-Net (`unet`)

```bash
python -m bapmos.external_baselines.unet.train_multiclass \
  --data_root data/prostate/pooled \
  --run_root runs/prostate/baselines --run_name unet_pooled_seed42 --seed 42 \
  --checkpoint-objective-metric ptv_kfold_msd

python -m bapmos.external_baselines.unet.run_test_inference \
  --checkpoint runs/prostate/baselines/unet/unet_pooled_seed42/best_checkpoint.pth \
  --output_dir inference_output/prostate/pooled/unet/unet_pooled_seed42
```

The trainer writes to `<run_root>/unet/<run_name>/` (default `--run_root` is
`runs/ExternalBaselines`).

### MedSAM (`medsam`)

MedSAM weights initialise SAM ViT-B; only the mask decoder is fine-tuned with box prompts.

```bash
python -m bapmos.external_baselines.medsam_init.verify_weights

python -m bapmos.external_baselines.medsam_init.train_decoder_boxes \
  --data_root data/prostate/pooled \
  --medsam_checkpoint models/medsam/medsam_vit_b.pth \
  --run_root runs/prostate/baselines --run_name medsam_pooled_seed42 --seed 42 \
  --checkpoint-objective-metric ptv_kfold_msd

python -m bapmos.external_baselines.medsam_init.run_test_inference \
  --checkpoint runs/prostate/baselines/medsam_init/medsam_pooled_seed42/best_checkpoint.pth \
  --output_dir inference_output/prostate/pooled/medsam/medsam_pooled_seed42
```

The trainer writes to `<run_root>/medsam_init/<run_name>/`.

### SAM (`box`)

The paper's SAM row is the SAM ViT-B mask decoder fine-tuned with box prompts. It shares its
trainer and run folder with the box-only rung of the fixed-schedule experiments.

```bash
python -m bapmos.multiorgan.train_sam_multiorgan_decoder_box \
  --data_root data/prostate/pooled \
  --run_root runs/prostate/legacy/box --run_name box_pooled_seed42 --seed 42 \
  --box_jitter_px 2 \
  --checkpoint-objective-metric ptv_kfold_msd

python -m bapmos.multiorgan.run_test_inference \
  --checkpoint runs/prostate/legacy/box/box_pooled_seed42/best_checkpoint.pth \
  --output_dir inference_output/prostate/pooled/box/box_pooled_seed42
```

Pass `--data_root` explicitly: its default is the single-site `case1` corpus. The trainer
writes to `<run_root>/<run_name>/` (default `--run_root` is `runs/<bundle>/Baseline`).
`--box_jitter_px 2` matches `configs/prostate/common.yaml`.

### nnU-Net (`nnunet`)

nnU-Net v2 runs in its own environment (`requirements_nnunet.txt`) with `src/` on
`PYTHONPATH`. Training uses `nnUNetTrainerBapMosProtocol`
(`src/bapmos/external_baselines/nnunet2d/nnunet_trainer_bap_mos.py`): stock Dice+CE loss,
flat learning rate, the repository's patient-level train/val split, and validation-MSD
checkpointing. nnU-Net v2 looks up trainer classes inside its own
`nnunetv2/training/nnUNetTrainer/` package, so link or copy that file there.

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results

# 1. Export train + val to nnU-Net raw format (dataset id 505 for pooled prostate)
python -m bapmos.external_baselines.nnunet2d.export_nnunet_dataset \
  --data_root data/prostate/pooled --output_parent "${nnUNet_raw}" \
  --dataset_id 505 --dataset_name BapMosPooledTrainVal

# 2. Plan and train (fold 0 = the repository's train/val split)
nnUNetv2_plan_and_preprocess -d 505 --verify_dataset_integrity
BAPMOS_NNUNET_DATA_ROOT=data/prostate/pooled \
NNUNET_MAX_EPOCHS=300 NNUNET_PATIENCE=40 \
  nnUNetv2_train 505 2d 0 -tr nnUNetTrainerBapMosProtocol
```

The trainer's defaults (`NNUNET_MAX_EPOCHS=100`, `NNUNET_PATIENCE=20`) follow the PFUS1
schedule. The prostate values above follow the paper's pooled-TRUS schedule (up to 300 epochs, patience 40). Checkpoint selection
uses fold-balanced validation MSD (`NNUNET_CHECKPOINT_OBJECTIVE=ptv_kfold_msd`, the default),
which needs `BAPMOS_NNUNET_DATA_ROOT`.

Pooled prostate keeps a separate test list per site, so predict and export one site at a time.
`--test_images_only` rewrites `imagesTs/` and the test part of `case_mapping.json` for that site:

```bash
DS=Dataset505_BapMosPooledTrainVal
for site in simulation case1 case2; do
  python -m bapmos.external_baselines.nnunet2d.export_nnunet_dataset \
    --data_root data/prostate/pooled --output_parent "${nnUNet_raw}" \
    --dataset_id 505 --dataset_name BapMosPooledTrainVal \
    --splits_subdir "site_tests/${site}" --test_images_only

  nnUNetv2_predict -i "${nnUNet_raw}/${DS}/imagesTs" -o "nnunet_pred/${site}" \
    -d 505 -c 2d -f 0 -tr nnUNetTrainerBapMosProtocol -chk checkpoint_best.pth

  python -m bapmos.external_baselines.nnunet2d.run_test_inference \
    --predictions_dir "nnunet_pred/${site}" \
    --case_mapping "${nnUNet_raw}/${DS}/case_mapping.json" \
    --data_root data/prostate/pooled \
    --splits-subdir "site_tests/${site}" \
    --output_dir inference_output/prostate/pooled/nnunet/nnunet_pooled_seed42 \
    --run-name pooled_seed42 --experiment-seed 42
done
```

Seeds: the nnU-Net trainer has no seed option, and nnU-Net v2 does not fix a training seed.
A replicate is another `nnUNetv2_train` run with its own `nnUNet_results` directory. At export,
name it with `--run-name pooled_seed43_rep2 --experiment-seed 43 --force-inference-output`
(and the matching output folder).

### Result tables

`bash scripts/populate_results_prostate.sh` also ingests the baseline exports
(`MAIN_ONLY=1` restricts it to BAP-MOS and the baselines).

---

## 4. Fixed-schedule and policy experiments (prostate)

These are the paper's fixed-schedule experiments (box only, point only, Box:Point and
Box:Point:Both ratios) and adaptive-policy experiments (UCB1 global and per organ, ε-greedy,
ε-decay). They use an earlier protocol: CE + Dice loss, negated validation MSD as reward, and
a fixed 20 px negative-point ring. See
[`experiments/prostate/legacy/README.md`](../experiments/prostate/legacy/README.md).

### Box only and point only

Box only is the SAM baseline above. Point only:

```bash
python -m bapmos.multiorgan.train_sam_multiorgan_decoder_points \
  --data_root data/prostate/pooled \
  --run_root runs/prostate/legacy/point --run_name point_pooled_seed42 --seed 42 \
  --checkpoint-objective-metric ptv_kfold_msd

python -m bapmos.multiorgan.run_test_inference \
  --checkpoint runs/prostate/legacy/point/point_pooled_seed42/best_checkpoint.pth \
  --output_dir inference_output/prostate/pooled/point/point_pooled_seed42
```

### Ratios and policies

`bapmos.legacy.optimization.trainer` reads one of the configs below and trains the experiment
named with `--experiment`. It ignores `run_root` in the YAML and writes to
`<--run_root>/<strategy>/<--run_name>/` (default `--run_root` is `runs/<bundle>/Optimization`).
Pass both flags.

| Config | Experiments (`--experiment`) | Strategy folder |
|--------|------------------------------|-----------------|
| `experiments/prostate/legacy/box_point/config.yaml` | `box10_point90`, `box50_point50`, `box90_point10` | `box_point` |
| `experiments/prostate/legacy/boxpoint_box_point/config.yaml` | `ratio_1_1_1_balanced`, `ratio_1_5_1_point_dom`, `ratio_2_3_5_prof`, `ratio_5_1_1_box_dom` | `boxpoint_box_point` |
| `experiments/prostate/legacy/policies/<policy>/config.yaml` | `<policy>_seed42`, `<policy>_seed43_rep2`, `<policy>_seed44_rep3` | `<policy>` |

`<policy>` is one of `ucb1_global`, `ucb1_per_organ`, `epsilon_greedy_per_organ`,
`epsilon_decay_per_organ`.

Ratio configs have one entry per ratio (seed 42). Set `PI_TRAIN_SEED` for the replicates:

```bash
CFG=experiments/prostate/legacy/box_point/config.yaml

python -m bapmos.legacy.optimization.trainer --config "${CFG}" \
  --experiment box50_point50 \
  --run_root runs/prostate/legacy --run_name box50_point50_seed42

PI_TRAIN_SEED=43 python -m bapmos.legacy.optimization.trainer --config "${CFG}" \
  --experiment box50_point50 \
  --run_root runs/prostate/legacy --run_name box50_point50_seed43_rep2

PI_TRAIN_SEED=44 python -m bapmos.legacy.optimization.trainer --config "${CFG}" \
  --experiment box50_point50 \
  --run_root runs/prostate/legacy --run_name box50_point50_seed44_rep3
```

Policy configs carry one entry per seed:

```bash
python -m bapmos.legacy.optimization.trainer \
  --config experiments/prostate/legacy/policies/ucb1_global/config.yaml \
  --experiment ucb1_global_seed42 \
  --run_root runs/prostate/legacy --run_name ucb1_global_seed42
```

Test export (`--method-slug` is the method id; add `--force-inference-output` for seeds 43/44):

```bash
python -m bapmos.legacy.optimization.run_test_inference \
  --checkpoint runs/prostate/legacy/box_point/box50_point50_seed42/best_checkpoint.pth \
  --output_dir inference_output/prostate/pooled/box_point/box50_point50_seed42 \
  --method-slug box_point
```

Result tables for this group only:

```bash
LADDER_ONLY=1 bash scripts/populate_results_prostate.sh
```

---

## 5. Bladder (PFUS1)

The paper reports BAP-MOS (MedSAM) on PFUS1 (`bapmos_medsam`). BAP-MOS (SAM) and the bladder
baselines are additional experiments, not reported in the paper.

### Data preparation

Build the bundle, the outer-loop search split, and the cached image embeddings
([`PREPROCESS.md`](PREPROCESS.md#bladder-pfus1)):

```bash
python -m bapmos.preprocess.bladder
python -m bapmos.preprocess.bladder.create_hpo_splits --data_root data/bladder/pfus1

python -m bapmos.preprocess.bladder.precompute_sam_embeddings \
  --init medsam --sam_checkpoint models/medsam/medsam_vit_b.pth \
  --output_dir data/bladder/pfus1/sam_embeddings/medsam_vit_b
python -m bapmos.preprocess.bladder.precompute_sam_embeddings \
  --init meta --sam_checkpoint models/sam_base/sam_vit_b_01ec64.pth \
  --output_dir data/bladder/pfus1/sam_embeddings/sam_vit_b
```

The inner loop reads the cached embeddings; the outer loop needs the search split.

### Outer loop (optional)

The committed `experiments/bladder/bapmos/inner_loop/{sam,medsam}/selected/pfus1.yaml` files hold
the paper's selected hyperparameters. To repeat the search:

```bash
python -m bapmos.hpo.study_runner init --hpo-suite bapmos_bo_medsam --dataset pfus1
HPO_SEQUENTIAL=1 python -m bapmos.hpo.study_runner run \
  --hpo-suite bapmos_bo_medsam --dataset pfus1 --n-trials 100
# optional, overwrites the committed selected/pfus1.yaml
python -m bapmos.hpo.study_runner export --hpo-suite bapmos_bo_medsam --dataset pfus1
```

Use `--hpo-suite bapmos_bo_sam` for the SAM backbone.

### Inner loop and test export

```bash
for exp in pfus1_seed42 pfus1_seed43_rep2 pfus1_seed44_rep3; do
  python -m bapmos.method.bap_mos_trainer \
    --version bapmos_medsam --dataset pfus1 --experiment "${exp}"
done

python -m bapmos.method.run_test_inference \
  --checkpoint runs/bladder/bapmos/inner_loop/medsam/pfus1_seed42/best_checkpoint.pth \
  --output_dir inference_output/bladder/pfus1/bapmos_medsam/pfus1_seed42
for exp in pfus1_seed43_rep2 pfus1_seed44_rep3; do
  python -m bapmos.method.run_test_inference \
    --checkpoint "runs/bladder/bapmos/inner_loop/medsam/${exp}/best_checkpoint.pth" \
    --output_dir "inference_output/bladder/pfus1/bapmos_medsam/${exp}" \
    --force-inference-output
done
```

For BAP-MOS (SAM) use `--version bapmos_sam`, checkpoints under
`runs/bladder/bapmos/inner_loop/sam/`, and output folder `bapmos_sam/`.

### Baselines (additional experiments)

Same trainers as prostate, with `--data_root data/bladder/pfus1`. U-Net and MedSAM pick the
PFUS1 schedule (100 epochs, patience 20, batch 128) from the data root.

```bash
python -m bapmos.external_baselines.unet.train_multiclass \
  --data_root data/bladder/pfus1 \
  --run_root runs/bladder/baselines --run_name unet_pfus1_seed42 --seed 42 \
  --checkpoint-objective-metric ptv_kfold_msd --checkpoint-objective-organ Bladder

python -m bapmos.external_baselines.medsam_init.train_decoder_boxes \
  --data_root data/bladder/pfus1 \
  --medsam_checkpoint models/medsam/medsam_vit_b.pth \
  --run_root runs/bladder/baselines --run_name medsam_pfus1_seed42 --seed 42 \
  --checkpoint-objective-metric ptv_kfold_msd --checkpoint-objective-organ Bladder

python -m bapmos.multiorgan.train_sam_multiorgan_decoder_box \
  --data_root data/bladder/pfus1 \
  --run_root runs/bladder/legacy/box --run_name box_pfus1_seed42 --seed 42 \
  --max_epochs 100 --patience 20 --batch_size 128 --box_jitter_px 10 \
  --checkpoint-objective-metric ptv_kfold_msd --checkpoint-objective-organ Bladder
```

The SAM-box trainer does not switch schedules by data root, so the flags above set the values
from `configs/bladder/common.yaml`. Test export uses the same `run_test_inference` modules as
prostate, with output folders `inference_output/bladder/pfus1/<method>/<run_name>/`
(`unet`, `medsam`, `box`).

nnU-Net on PFUS1 uses dataset id 503 and the default (PFUS1) schedule:

```bash
python -m bapmos.external_baselines.nnunet2d.export_nnunet_dataset \
  --data_root data/bladder/pfus1 --output_parent "${nnUNet_raw}" \
  --dataset_id 503 --dataset_name BapMosPfus1TrainVal
nnUNetv2_plan_and_preprocess -d 503 --verify_dataset_integrity
BAPMOS_NNUNET_DATA_ROOT=data/bladder/pfus1 NNUNET_CHECKPOINT_OBJECTIVE_ORGAN=Bladder \
  nnUNetv2_train 503 2d 0 -tr nnUNetTrainerBapMosProtocol

python -m bapmos.external_baselines.nnunet2d.export_nnunet_dataset \
  --data_root data/bladder/pfus1 --output_parent "${nnUNet_raw}" \
  --dataset_id 503 --dataset_name BapMosPfus1TrainVal --test_images_only
nnUNetv2_predict -i "${nnUNet_raw}/Dataset503_BapMosPfus1TrainVal/imagesTs" -o nnunet_pred/pfus1 \
  -d 503 -c 2d -f 0 -tr nnUNetTrainerBapMosProtocol -chk checkpoint_best.pth
python -m bapmos.external_baselines.nnunet2d.run_test_inference \
  --predictions_dir nnunet_pred/pfus1 \
  --case_mapping "${nnUNet_raw}/Dataset503_BapMosPfus1TrainVal/case_mapping.json" \
  --data_root data/bladder/pfus1 \
  --output_dir inference_output/bladder/pfus1/nnunet/nnunet_pfus1_seed42 \
  --run-name pfus1_seed42 --experiment-seed 42
```

### Result tables

```bash
bash scripts/populate_results_bladder.sh
```

The script ingests every run folder under `inference_output/bladder/pfus1/<method>/` whose name
ends in `_seed42`, `_seed43_rep2` or `_seed44_rep3`, skips missing runs with a warning, and
builds `results/bladder/pfus1/combined/`. The manual form is in
[`RESULTS.md`](RESULTS.md#manual-collation).

---

## 6. Batch systems

The repository has no scheduler scripts. Wrap any command above in your own job script and run
it from the repository root with `src/` on `PYTHONPATH`.

---

## 7. Reproduction checklist

- [ ] Three training seeds (42, 43, 44) for each reported method
- [ ] Test export for all three seeds under `inference_output/…/<method>/<run_name>/`
      (seeds 43 and 44 with `--force-inference-output`)
- [ ] `populate_results_prostate.sh` / `populate_results_bladder.sh`, or manual
      `collate_seeds` ingest and `build`
- [ ] Prostate numbers read from `results/prostate/pooled/combined/` rows with `site=pooled`
