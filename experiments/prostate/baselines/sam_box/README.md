# Baseline: SAM ViT-B + box prompts (decoder fine-tune).

**Do not use a separate ``runs/*/baselines/sam_box/`` tree.**  
This recipe is the same as **legacy box-only**:

- Entry: ``python -m bapmos.multiorgan.train_sam_multiorgan_decoder_box``
- Runs: ``runs/prostate/legacy/box/`` (bladder: ``runs/bladder/legacy/box/``)
- Test export: ``python -m bapmos.multiorgan.run_test_inference`` — see ``docs/RUNNING.md``

Loss: regional CE+Dice only (not Kervadec). Weights: ``models/sam_base/sam_vit_b_01ec64.pth``.
Seeds 42 / 43 / 44 — see ``docs/SEEDS.md``.
