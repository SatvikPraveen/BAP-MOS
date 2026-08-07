# Baseline: SAM ViT-B + box prompts (decoder fine-tune).

**Do not use ``runs/bladder/baselines/sam_box/``.** Same recipe as legacy box:

- Runs: ``runs/bladder/legacy/box/``
- Test export: ``python -m bapmos.multiorgan.run_test_inference`` — see ``docs/RUNNING.md``

Entry: ``python -m bapmos.multiorgan.train_sam_multiorgan_decoder_box``. CE+Dice only.
