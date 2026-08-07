# Model weights

Foundation checkpoints are **not** shipped in git. Place them under `BAPMOS/models/`
(canonical standalone layout).

Empty layout dirs (`models/`, `models/sam_base/`, `models/medsam/`) are committed via
`.gitkeep`; `*.pth` files are gitignored.

Use of SAM and MedSAM weights follows the respective upstream licenses; this repo does
not redistribute those model files.

## Cluster / batch systems

This package does not ship scheduler scripts. Wrap any `python -m` command from
`docs/RUNNING.md` in your own batch system. If you add `#SBATCH --output` / `--error`
paths, create their **parent directories before** the job starts — schedulers like Slurm
open those files before any `mkdir` in the batch script runs.

## SAM (ViT-B)

- **Path:** `BAPMOS/models/sam_base/sam_vit_b_01ec64.pth`
- **Source:** [Meta Segment Anything — model checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)
- **Direct download:** https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

```bash
# from BAPMOS/
mkdir -p models/sam_base
curl -L -o models/sam_base/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## MedSAM (ViT-B)

- **Path:** `BAPMOS/models/medsam/medsam_vit_b.pth`
- **Source:** [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM)
- **Preferred download (Zenodo):** https://zenodo.org/records/10689643  
  Direct file: https://zenodo.org/records/10689643/files/medsam_vit_b.pth
- **Alternate:** [Google Drive folder](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN) (official MedSAM README mirror)

```bash
# from BAPMOS/
mkdir -p models/medsam
curl -L -o models/medsam/medsam_vit_b.pth \
  https://zenodo.org/records/10689643/files/medsam_vit_b.pth
```

### MedSAM-init baseline overlap check

The external baseline ``medsam_init`` builds SAM ViT-B from ``sam_vit_b_01ec64.pth``, overlays
MedSAM weights, then fine-tunes the mask decoder only. On training start, ``config.json`` records
``medsam_missing_keys``, ``medsam_unexpected_keys``, ``medsam_overlap_ratio``, and related fields.

For canonical ``medsam_vit_b.pth`` (Zenodo), expect **very few** missing keys (typically 0–10)
and overlap ratio **≥ 95%**. Loads outside those bounds fail by default.

**Quick preflight** (no training run required):

```bash
python -m bapmos.external_baselines.medsam_init.verify_weights
```

Override thresholds via ``MEDSAM_MAX_MISSING_KEYS`` (default 50) or
``MEDSAM_MIN_OVERLAP_RATIO`` (default 0.95).

MedSAM encoder weights are **re-applied** on training resume and on standalone
``run_test_inference`` so test evaluation uses the same backbone as training.

## Path resolution

Relative model paths are resolved under `BAPMOS/models/`.
For a standalone checkout, place all required pretrained weights there.

## Trained run checkpoints

Fine-tuned / production checkpoints (`best_checkpoint.pth`, etc.) live under `runs/`, not under
`models/`. Inference exports go to `BAPMOS/inference_output/`.
