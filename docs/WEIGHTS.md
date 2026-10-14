# Model weights

Foundation-model checkpoints are not in the repository. Place them under `models/` in the
repository root. The empty folders `models/sam_base/` and `models/medsam/` are committed;
`*.pth` files are git-ignored.

SAM and MedSAM weights are covered by their upstream licences. This repository does not
redistribute them.

## SAM (ViT-B)

- Path: `models/sam_base/sam_vit_b_01ec64.pth`
- Source: [Segment Anything model checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)

```bash
mkdir -p models/sam_base
curl -L -o models/sam_base/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## MedSAM (ViT-B)

- Path: `models/medsam/medsam_vit_b.pth`
- Source: [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM)
- Download: [Zenodo record 10689643](https://zenodo.org/records/10689643), or the
  [Google Drive folder](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN)
  linked from the MedSAM README

```bash
mkdir -p models/medsam
curl -L -o models/medsam/medsam_vit_b.pth \
  https://zenodo.org/records/10689643/files/medsam_vit_b.pth
```

### MedSAM weight check

The MedSAM baseline builds SAM ViT-B from `sam_vit_b_01ec64.pth` and loads the MedSAM weights on
top. Check the overlap before training:

```bash
python -m bapmos.external_baselines.medsam_init.verify_weights
```

Loading fails if more than `MEDSAM_MAX_MISSING_KEYS` (default 50) keys are missing, or if the
overlap ratio is below `MEDSAM_MIN_OVERLAP_RATIO` (default 0.95). The canonical Zenodo file
typically misses 0–10 keys. Both thresholds can be set as environment variables, or passed to
`verify_weights` as `--max-missing-keys` and `--min-overlap-ratio`.

The MedSAM baseline records `medsam_missing_keys`, `medsam_unexpected_keys` and
`medsam_overlap_ratio` in its run's `config.json`. MedSAM weights are re-applied on resume and in
`run_test_inference`, so test evaluation uses the training backbone.

## Paths

Relative model paths in configs and CLI flags resolve under `models/`. Fine-tuned checkpoints
(`best_checkpoint.pth`, `last_checkpoint.pth`) are written under `runs/`, and test exports under
`inference_output/` ([`RUNTIME_LAYOUT.md`](RUNTIME_LAYOUT.md)).
