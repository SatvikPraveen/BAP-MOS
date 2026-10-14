# Preprocessing

Neither dataset is distributed with this repository. Place the source data as described below,
or symlink existing processed folders into `data/`. Run every command from the repository root
with `src/` on `PYTHONPATH`.

## Prostate TRUS

The prostate corpus has three sites: a phantom simulation and two clinical cohorts (`case1`,
`case2`). Training uses the pooled corpus built from all three.

### Input layout

DICOM inputs (MR slices and RTSTRUCT), read by `run_rtstruct_masks`:

```text
data/real_data/Case 1/Dicom/      # case1 (Case1/ or dcm_files/ also accepted)
data/real_data/Case 2/Dicom/      # case2
data/simulation_data/dicom/       # simulation
```

Per-site bundles, written by the steps below (or symlinked):

```text
data/prostate/{simulation,case1,case2}/
  images/                         # slice PNGs (or <site>_dicom_png/)
  masks/combined_masks/           # *_combined_mask.png
  splits_stratified/              # train.txt, val.txt, test.txt
```

Pooled corpus:

```text
data/prostate/pooled/
  images/
  masks/combined_masks/
  splits_stratified/              # train + val from all sites
  site_tests/<site>/test.txt      # one test list per site
  spacing_contract.json
```

### Steps

```bash
# 1. DICOM -> PNG slices and RTSTRUCT -> multiclass masks, per site
python -m bapmos.preprocess.prostate.run_rtstruct_masks --case all

# 2. Stratified train / val / test splits per site (seed 42)
python -m bapmos.preprocess.prostate.create_stratified_splits --all --seed 42

# 3. Build data/prostate/pooled/ (clinical files are symlinked; use --copy to copy)
python -m bapmos.preprocess.prostate
```

`run_rtstruct_masks --case` also takes `case1`, `case2` or `simulation`.
`create_stratified_splits --dataset` takes `simulation`, `case_1`, `case_2` or a path.
`python -m bapmos.preprocess.prostate --dry-run` prints the pooling plan without writing.
The modules live in `src/bapmos/preprocess/prostate/`.

## Bladder (PFUS1)

### Input layout

Raw frames:

```text
data/bladder/pfus1_raw/
  Pxxx/
    frame_*.png
    frame_*.json
```

Processed bundle:

```text
data/bladder/pfus1/
  masks/
  splits_patient_70_15_15_seed42/           # train / val / test, split by patient
  splits_hpo_tpe_patient_seed4242_p20/      # outer-loop search split
  sam_embeddings/{sam_vit_b,medsam_vit_b}/  # cached image embeddings
```

### Steps

```bash
# 1. JSON polygons -> masks, patient-level splits, label check
python -m bapmos.preprocess.bladder

# 2. Outer-loop search split (needed before the bladder outer loop)
python -m bapmos.preprocess.bladder.create_hpo_splits --data_root data/bladder/pfus1

# 3. Cached image embeddings (needed by the bladder inner loop)
python -m bapmos.preprocess.bladder.precompute_sam_embeddings \
  --init meta --sam_checkpoint models/sam_base/sam_vit_b_01ec64.pth \
  --output_dir data/bladder/pfus1/sam_embeddings/sam_vit_b
python -m bapmos.preprocess.bladder.precompute_sam_embeddings \
  --init medsam --sam_checkpoint models/medsam/medsam_vit_b.pth \
  --output_dir data/bladder/pfus1/sam_embeddings/medsam_vit_b
```

Step 1 runs `convert`, `splits` and `verify`; `--step` runs one of them and `--dry-run` prints
the plan.

Step 2 takes 20% of the 77 training patients (seed 4242) as the search cohort and holds out 20%
of that cohort as search validation. The original validation and test patients are not touched.

Step 3: the frozen image encoder makes the bladder inner loop read cached embeddings
(`use_cached_image_embeddings: true` in `experiments/bladder/bapmos/inner_loop/{sam,medsam}/version.yaml`).
The output folders match `image_embedding_dir` in those files. By default the script encodes the
`train`, `val` and `test` lists of `splits_patient_70_15_15_seed42` on `cuda` and stores
`float16`. With `--init medsam`, the SAM ViT-B architecture comes from
`--architecture_checkpoint` (default `models/sam_base/sam_vit_b_01ec64.pth`).

The bladder checkpoint and search objective use the `Bladder` organ
(`checkpoint_objective.objective_organ`). The modules live in `src/bapmos/preprocess/bladder/`.

## Quality checks (optional)

```bash
python -m bapmos.preprocess.delineation overlays --dataset case_1 --splits-subdir splits_stratified
python -m bapmos.preprocess.delineation summarize
python -m bapmos.preprocess.delineation organ-presence --all
python -m bapmos.preprocess.bladder.visualize_samples   # PFUS1 overlays
```

Outputs go under `data/prostate/` (`qa_overlays/`, `slice_organ_presence/`,
`rt_rs_rp_mask_summary/`).

The test-export folder layout is described in [`RESULTS.md`](RESULTS.md#test-export-layout).
