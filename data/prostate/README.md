# Prostate TRUS data

The prostate data are not distributed with this repository. Do not commit them.

Training reads the pooled corpus in `data/prostate/pooled/`, built from three sites:

```text
data/prostate/
  simulation/          # phantom simulation
  case1/               # clinical cohort 1
  case2/               # clinical cohort 2
  pooled/              # built from the three sites
```

Each site folder holds:

```text
images/                # slice PNGs (or <site>_dicom_png/)
masks/combined_masks/  # *_combined_mask.png
splits_stratified/     # train.txt, val.txt, test.txt
```

The pooled folder holds:

```text
images/
masks/combined_masks/
splits_stratified/     # train + val from all sites
site_tests/<site>/test.txt
spacing_contract.json
```

Build the site folders from DICOM and RTSTRUCT, then the pooled corpus:

```bash
python -m bapmos.preprocess.prostate.run_rtstruct_masks --case all
python -m bapmos.preprocess.prostate.create_stratified_splits --all --seed 42
python -m bapmos.preprocess.prostate
```

DICOM input locations and the full steps are in
[`docs/PREPROCESS.md`](../../docs/PREPROCESS.md#prostate-trus).

If you already have a pooled corpus, link it instead (from `data/prostate/`):

```bash
ln -s /path/to/existing/pooled pooled
```
