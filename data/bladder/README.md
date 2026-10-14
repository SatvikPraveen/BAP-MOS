# Bladder PFUS1 data

PFUS1 ([Solís-Martín et al., 2025](https://www.cs.us.es/~dsolis/en/pfus1.html)) is not
distributed with this repository. Do not commit it.

```text
data/bladder/
  pfus1_raw/                                  # raw frames
    Pxxx/frame_*.png
    Pxxx/frame_*.json
  pfus1/                                      # processed bundle
    masks/
    splits_patient_70_15_15_seed42/
    splits_hpo_tpe_patient_seed4242_p20/      # outer-loop search split
    sam_embeddings/{sam_vit_b,medsam_vit_b}/  # cached image embeddings
```

Build the bundle, the search split and the embeddings:

```bash
python -m bapmos.preprocess.bladder
python -m bapmos.preprocess.bladder.create_hpo_splits --data_root data/bladder/pfus1
python -m bapmos.preprocess.bladder.precompute_sam_embeddings \
  --init meta --sam_checkpoint models/sam_base/sam_vit_b_01ec64.pth \
  --output_dir data/bladder/pfus1/sam_embeddings/sam_vit_b
python -m bapmos.preprocess.bladder.precompute_sam_embeddings \
  --init medsam --sam_checkpoint models/medsam/medsam_vit_b.pth \
  --output_dir data/bladder/pfus1/sam_embeddings/medsam_vit_b
```

Details: [`docs/PREPROCESS.md`](../../docs/PREPROCESS.md#bladder-pfus1).

If you already have the data elsewhere, link it instead (from `data/bladder/`):

```bash
ln -s /path/to/existing/pfus1_raw pfus1_raw
ln -s /path/to/existing/pfus1 pfus1
```
