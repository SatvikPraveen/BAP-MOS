# Running experiments (scheduler-agnostic)

This repository contains the **BAP-MOS library, configs, and experiment definitions**.
It does **not** ship cluster job scripts (Slurm, PBS, etc.). Every step below is a
`python -m …` command runnable interactively or wrapped by your own batch system.

**Prerequisites:** from `BAPMOS/`, `export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"`
(or use `./scripts/bapmos.sh` for preprocess/tests).

Artifact layout: `docs/RUNTIME_LAYOUT.md`. Experiment ladder: `docs/EXPERIMENT_LADDER.md`.
Results collation: `docs/RESULTS.md`. Seeds: `docs/SEEDS.md` (42, 43, 44).

---

## 1. Setup

```bash
cd BAPMOS
pip install -r requirements.txt
pip install -r requirements-hpo.txt    # outer-loop search
pip install -r requirements_dev.txt    # pytest (+ optional notebook extras)
# CPU-only: pip install -r requirements_cpu.txt
# nnU-Net baseline (dedicated env): pip install -r requirements_nnunet.txt
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

# Weights → models/ (docs/WEIGHTS.md)
# Data → data/prostate/pooled, data/bladder/pfus1 (data/*/README.md)

./scripts/bapmos.sh preprocess-prostate --dry-run
```

There is **no** `pyproject.toml` — run with `PYTHONPATH=src` (or `./scripts/bapmos.sh`).
Use `--no-wandb` or `wandb.enabled: false` if you omit W&B.

| File | Role |
|------|------|
| `requirements.txt` | Core train / preprocess / inference / W&B |
| `requirements_cpu.txt` | Same as core + PyTorch CPU index |
| `requirements_dev.txt` | pytest + Jupyter / Plotly / PyVista extras |
| `requirements-hpo.txt` | Optuna outer-loop search |
| `requirements_nnunet.txt` | nnU-Net v2 (dedicated env recommended) |

---

## 2. Main path — prostate BAP-MOS

### Outer loop (hyperparameter search, seed 42 only)

```bash
python -m bapmos.hpo.study_runner init --hpo-suite bapmos_bo --dataset pooled
python -m bapmos.hpo.study_runner run --hpo-suite bapmos_bo --dataset pooled --n-trials <N>
```

Search ablations: suites `bapmos_bo_random`, `bapmos_bo_greedy`, `bapmos_bo_heuristic`.
MedSAM init: `bapmos_bo_medsam_pooled`. See `docs/SEARCH_METHODS.md`.

### Export selected hyperparameters

After search, export best trial into `experiments/prostate/bapmos/inner_loop/selected/<method>/pooled.yaml`
(`generation` must be ≥ 1 — placeholder `generation: 0` files are refused by `compose_config`).

```bash
python -m bapmos.hpo.study_runner export \
  --hpo-suite bapmos_bo --dataset pooled
```

Details: `docs/INNER_OUTER_LOOP.md`.

### Inner loop (production, three seeds)

```bash
for exp in pooled_seed42 pooled_seed43_rep2 pooled_seed44_rep3; do
  python -m bapmos.method.bap_mos_trainer \
    --version bapmos --dataset pooled --experiment "${exp}"
done
```

Checkpoints: `runs/prostate/bapmos/inner_loop/<search_method>/${exp}/best_checkpoint.pth`.

### Test inference export (all three seeds)

Stratified export writes metrics + panels under `inference_output/`.
By default only seed 42 is allowed; seeds 43/44 need `--force-inference-output`.
For three-seed mean ± std tables, export **every** seed to a per-seed directory:

```bash
CKPT_ROOT=runs/prostate/bapmos/inner_loop/tpe
OUT_ROOT=inference_output/prostate/pooled/bapmos

# Seed 42 (force optional; kept here for a uniform layout)
python -m bapmos.method.run_test_inference \
  --checkpoint "${CKPT_ROOT}/pooled_seed42/best_checkpoint.pth" \
  --output_dir "${OUT_ROOT}/pooled_seed42"

# Seeds 43 / 44 (force required)
for exp in pooled_seed43_rep2 pooled_seed44_rep3; do
  python -m bapmos.method.run_test_inference \
    --checkpoint "${CKPT_ROOT}/${exp}/best_checkpoint.pth" \
    --output_dir "${OUT_ROOT}/${exp}" \
    --force-inference-output
done
```

Each export contains per-site folders (`simulation/`, `case1/`, `case2/`) with metrics on the
full test split (panels for the first 10 slices). See `docs/SEEDS.md`.

### Results tables

Ingest **from the three `inference_output/` exports above**, then build mean ± std:

```bash
for seed_run in pooled_seed42 pooled_seed43_rep2 pooled_seed44_rep3; do
  python -m bapmos.results.collate_seeds ingest-run \
    --corpus prostate --method bapmos --run-name "${seed_run}" \
    --run-dir "inference_output/prostate/pooled/bapmos/${seed_run}"
done

python -m bapmos.results.collate_seeds build --corpus prostate
```

Or batch-ingest legacy + main exports once those per-seed dirs exist:

```bash
bash scripts/populate_results_prostate.sh
```

---

## 3. Baselines — prostate

| Baseline | Train | Test export |
|----------|-------|-------------|
| UNet | `python -m bapmos.external_baselines.unet.train_multiclass …` | `python -m bapmos.external_baselines.unet.run_test_inference …` |
| MedSAM-init+box | `python -m bapmos.external_baselines.medsam_init.train_decoder_boxes …` | `python -m bapmos.external_baselines.medsam_init.run_test_inference …` |
| nnU-Net | nnU-Net v2 pipeline (`requirements_nnunet.txt`) | `python -m bapmos.external_baselines.nnunet2d.run_test_inference …` |
| SAM box | `python -m bapmos.multiorgan.train_sam_multiorgan_decoder_box …` | `python -m bapmos.multiorgan.run_test_inference …` |

Run roots: `runs/prostate/baselines/{unet,medsam_init,nnunet}/` (SAM box ≡ `runs/prostate/legacy/box/`).

Three seeds per baseline. See `experiments/prostate/baselines/*/README.md`.

---

## 4. Legacy ladder — prostate (historical)

Not required for main BAP-MOS tables. Protocol differs (CE+Dice, MSD-only reward).

| Rung | Trainer | Config |
|------|---------|--------|
| box / point | `bapmos.multiorgan.train_sam_multiorgan_decoder_{box,points}` | `experiments/prostate/legacy/{box,point}/config.yaml` |
| box:point ratios | `bapmos.legacy.optimization.trainer` | `experiments/prostate/legacy/box_point/config.yaml` |
| box/point/both | same | `experiments/prostate/legacy/boxpoint_box_point/config.yaml` |
| policies | same | `experiments/prostate/legacy/policies/*/config.yaml` |

Example:

```bash
python -m bapmos.legacy.optimization.trainer \
  --config experiments/prostate/legacy/box_point/config.yaml \
  --experiment box50_point50_seed42
```

Run roots (canonical): `runs/prostate/legacy/…` — see `docs/RUNTIME_LAYOUT.md`.

### Legacy test inference

```bash
# box / point
python -m bapmos.multiorgan.run_test_inference \
  --checkpoint runs/prostate/legacy/box/box_pooled_seed42/best_checkpoint.pth \
  --output_dir inference_output/prostate/pooled/box/box_pooled_seed42

# ratios / policies
python -m bapmos.legacy.optimization.run_test_inference \
  --checkpoint runs/prostate/legacy/box_point/box50_point50_seed42/best_checkpoint.pth \
  --output_dir inference_output/prostate/pooled/box_point/box50_point50_seed42 \
  --method-slug box_point
```

### Legacy results batch

```bash
LADDER_ONLY=1 bash scripts/populate_results_prostate.sh
```

---

## 5. Bladder (PFUS1)

Same outer → export → inner pattern with suites `bapmos_bo_sam` / `bapmos_bo_medsam`
and `--version bapmos_sam` / `bapmos_medsam` (`docs/EXPERIMENT_LADDER.md`).

Run names: `pfus1_seed42`, `pfus1_seed43_rep2`, `pfus1_seed44_rep3`.

---

## 6. Cluster / batch systems

Wrap any command above in your scheduler. This repo intentionally stays scheduler-agnostic.

If you maintain a **separate harness** (e.g. Slurm scripts that `cd` into `BAPMOS/` and
set `PYTHONPATH`), keep it outside this repository. The harness should:

1. Call the same `python -m` entrypoints documented here.
2. Use the same `runs/`, `inference_output/`, `results/` paths as `docs/RUNTIME_LAYOUT.md`.
3. Optionally call `bash scripts/populate_results_prostate.sh` after inference batches complete.

---

## 7. Checklist before publishing results

- [ ] Three training seeds (42, 43, 44) for each reported experiment
- [ ] Stratified exports for all three seeds under `BAPMOS/inference_output/.../<run_name>/` (43/44 with `--force-inference-output`)
- [ ] `collate_seeds ingest-run` for all three seeds from those dirs, then `build` (or `populate_results_prostate.sh`)
- [ ] Aggregate numbers from `results/.../combined/` with `site=pooled`
- [ ] Inner loop only after real `selected/*.yaml` export (`generation ≥ 1`)
