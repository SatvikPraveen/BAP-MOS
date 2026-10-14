# Experiment ladder

Experiment definitions live under `experiments/<corpus>/`. Every reported method is trained with
three seeds (42, 43, 44). Commands: [`RUNNING.md`](RUNNING.md).

## Prostate (`experiments/prostate/`)

| Folder | What it defines | In the paper |
|--------|-----------------|--------------|
| `bapmos/outer_loop/tpe/` → `bapmos/inner_loop/` | BAP-MOS (SAM), TPE search | Main table |
| `bapmos/outer_loop/medsam/` → `bapmos/inner_loop/medsam/` | BAP-MOS (MedSAM) | Main table |
| `bapmos/outer_loop/{random,greedy,heuristic}/` | Other search strategies, same 5-D space | Search-strategy comparison |
| `baselines/{unet,nnunet,medsam}/` | U-Net, nnU-Net, MedSAM baselines | Main table |
| `baselines/sam_box/` | SAM baseline; trained as `legacy/box` | Main table (SAM row) |
| `legacy/box/` | Box-only prompts | Main table (SAM row) |
| `legacy/point/` | Point-only prompts | Run alongside the fixed-schedule experiments |
| `legacy/box_point/` | Box:Point = 10/90, 50/50, 90/10 | Fixed-schedule table |
| `legacy/boxpoint_box_point/` | Box:Point:Both = 1:1:1, 1:5:1, 2:3:5, 5:1:1 | Fixed-schedule table |
| `legacy/policies/` | UCB1 global and per organ, ε-greedy, ε-decay | Adaptive policy variants |

The `legacy/` experiments use an earlier protocol (CE + Dice loss, negated-MSD reward, fixed
20 px ring); see [`experiments/prostate/legacy/README.md`](../experiments/prostate/legacy/README.md).

## Bladder (`experiments/bladder/`)

| Folder | What it defines | In the paper |
|--------|-----------------|--------------|
| `bapmos/outer_loop/medsam/` → `bapmos/inner_loop/medsam/` | BAP-MOS (MedSAM) | PFUS1 table |
| `bapmos/outer_loop/sam/` → `bapmos/inner_loop/sam/` | BAP-MOS (SAM) | Additional experiment, not reported in the paper |
| `baselines/{unet,nnunet,sam_box,medsam}/` | Baselines | Additional experiments, not reported in the paper |
| `legacy/box/`, `legacy/point/` | Box-only (also the `sam_box` baseline) and point-only prompts | Additional experiments, not reported in the paper |

Bladder has no search-strategy comparison and no policy experiments.

Protocol details and the two loops: [`INNER_OUTER_LOOP.md`](INNER_OUTER_LOOP.md). Search
methods: [`SEARCH_METHODS.md`](SEARCH_METHODS.md).
