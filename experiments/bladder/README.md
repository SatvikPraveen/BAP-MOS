# Bladder (PFUS1) experiments

PFUS1 experiments use `data/bladder/pfus1/`, defaults from `configs/bladder/common.yaml`
(max 100 epochs, patience 20), the `Bladder` organ for checkpointing and search, and three seeds
(42, 43, 44).

| Folder | Contents | In the paper |
|--------|----------|--------------|
| `bapmos/outer_loop/medsam/`, `bapmos/inner_loop/medsam/` | BAP-MOS (MedSAM): suites `bapmos_bo_medsam` / `bapmos_medsam` | PFUS1 table |
| `bapmos/outer_loop/sam/`, `bapmos/inner_loop/sam/` | BAP-MOS (SAM): suites `bapmos_bo_sam` / `bapmos_sam` | Additional experiment, not reported in the paper |
| `baselines/{unet,nnunet,sam_box,medsam}/` | Baselines | Additional experiments, not reported in the paper |
| `legacy/box/`, `legacy/point/` | Box-only and point-only prompts (earlier protocol) | Additional experiments, not reported in the paper |

The FPN row of the PFUS1 table is quoted from García et al. (2024) and is not reproduced here.
Bladder has no search-strategy comparison and no policy experiments.

Commands: [`docs/RUNNING.md`](../../docs/RUNNING.md#5-bladder-pfus1). Protocol:
[`docs/INNER_OUTER_LOOP.md`](../../docs/INNER_OUTER_LOOP.md).
