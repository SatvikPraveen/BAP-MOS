# Bladder baselines — UNet / nnU-Net / SAM-box / MedSAM

Mirror of the prostate main-paper baselines, using `configs/bladder/common.yaml`
(Bladder MSD objective) and `data/bladder/pfus1/`.

**Loss:** all of these use regional CE+Dice (or nnU-Net's own Dice+CE).
**Kervadec is not used** on baselines — only on the main BAP-MOS method path.

Method-specific notes and CLIs: see `unet/`, `nnunet/`, `sam_box/`, `medsam/`
in this directory (and the prostate counterparts under
`experiments/prostate/baselines/` for longer CLI examples).

**MedSAM-init** (`medsam/`): not frozen MedSAM — MedSAM encoder overlay + decoder-only
box fine-tuning. Run `python -m bapmos.external_baselines.medsam_init.verify_weights`
before first training to confirm weight overlap.

Entry modules: `bapmos.external_baselines.*` and `bapmos.multiorgan.*`.

Three seeds: 42 / 43 / 44 — see `docs/SEEDS.md` and `docs/EXPERIMENT_LADDER.md`.
