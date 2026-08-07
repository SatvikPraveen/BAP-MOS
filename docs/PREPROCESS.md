# Preprocessing

## Required input layout

Real data are **not** shipped. Place or symlink corpora under `data/` as below
(see also `data/prostate/README.md` and `data/bladder/README.md`).

### Prostate (pooled)

Expected under `data/prostate/pooled/` (after build, or via symlink):

```text
data/prostate/pooled/
  images/
  masks/
  splits_stratified/
  spacing_contract.json
  site_tests/
```

Upstream site corpora (simulation / case1 / case2) with RTSTRUCT→mask bundles are required
before building the pooled tree, or generate masks with `run_rtstruct_masks` from RS/RP inputs.

### Bladder (PFUS1)

Raw frames (preferred):

```text
data/bladder/pfus1_raw/
  Pxxx/
    frame_*.png
    frame_*.json
```

Processed bundle (after preprocess, or via symlink):

```text
data/bladder/pfus1/
  masks/
  splits_*/          # e.g. splits_patient_70_15_15_seed42/
  # label registry / report metadata as produced by the bladder pipeline
```

## Prostate (pooled)

```bash
python -m bapmos.preprocess.prostate --help
# Builds data/prostate/pooled/ from site corpora (simulation + case1 + case2)
python -m bapmos.preprocess.prostate.create_stratified_splits --dataset case_1 --seed 42
python -m bapmos.preprocess.prostate.run_rtstruct_masks --case case1
```

Package layout: `src/bapmos/preprocess/prostate/` (`build_pooled`, RTSTRUCT export, stratified splits, spacing).

The pooled corpus is always written/read at `BAPMOS/data/prostate/pooled/`.
Symlink an existing pooled tree into that path if you prefer not to rebuild.

## Bladder (PFUS1)

```bash
python -m bapmos.preprocess.bladder --help
# convert JSON polygons → masks; patient splits → data/bladder/pfus1/
```

Package layout: `src/bapmos/preprocess/bladder/`. Compatibility shims remain under `bapmos.pfus1.*`.

Checkpoint organ for bladder runs: `Bladder` (`checkpoint_objective.objective_organ`).

## Delineation / mask overlaps

```bash
python -m bapmos.preprocess.delineation --help
python -m bapmos.preprocess.delineation overlays --dataset case_1 --splits-subdir splits_stratified
python -m bapmos.preprocess.delineation summarize
python -m bapmos.preprocess.delineation organ-presence --all
# PFUS1 overlays:
python -m bapmos.preprocess.bladder.visualize_samples
```

QA outputs default under `data/prostate/` (`qa_overlays/`, `slice_organ_presence/`, `rt_rs_rp_mask_summary/`).

## Inference export layout

Test-split exports (via `bapmos.inference_output` helpers, e.g. `python -m bapmos.method.run_test_inference` and baseline `run_test_inference` modules) write under `BAPMOS/inference_output/`:

```text
predictions/                 # sibling of visualizations/
  multiclass/*_pred_ids.png
  *_pred.png
visualizations/
  *_viz.png|.pdf             # prediction overlays
  difference/*_diff.png|.pdf # difference overlays (TP/FP/FN)
metrics/
```

Implemented in `src/bapmos/inference_output/export_bundle.py` (`export_stratified_test_bundle`).
