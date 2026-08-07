# Experiment ladder

**Main path (reproduce paper method + baselines):**

- **Prostate:** `experiments/prostate/bapmos/outer_loop/tpe/` (main) plus search ablations `outer_loop/{random,greedy,heuristic}/` → `inner_loop/selected/{…}/pooled.yaml`
- **Bladder:** `experiments/bladder/bapmos/outer_loop/{sam,medsam}/` → `inner_loop/{sam,medsam}/selected/pfus1.yaml`
- `experiments/*/baselines/` — UNet, nnU-Net, SAM-box, MedSAM (SAM-box uses the same trainer as `legacy/box` — store runs under `runs/*/legacy/box/`, not a second `baselines/sam_box/` tree)

**Search-method ablations (same BAP-MOS protocol, different HPO):** TPE (main), random, greedy, heuristic — see `docs/SEARCH_METHODS.md`.

Defined for **prostate** under `experiments/prostate/bapmos/outer_loop/{tpe,random,greedy,heuristic}/version.yaml` and `inner_loop/selected/{…}/pooled.yaml`.

**Bladder** keeps the main BAP-MOS protocol with a **backbone** split (not the prostate search-method family):

- `experiments/bladder/bapmos/outer_loop/sam/` — suite `bapmos_bo_sam`
- `experiments/bladder/bapmos/outer_loop/medsam/` — suite `bapmos_bo_medsam`
- Export → `experiments/bladder/bapmos/inner_loop/{sam,medsam}/selected/pfus1.yaml`
- Inner production: max_epochs 100 / patience 20; three seeds `pfus1_seed42` / `pfus1_seed43_rep2` / `pfus1_seed44_rep3`

**Historical only (not required for main tables):**

- `experiments/*/legacy/` — earlier prompt comparisons (box / point / ratios / policies on **prostate**; bladder ships box/point only — policies are not implemented there)
  (older MSD-only reward, fixed ring, CE+Dice — see each `legacy/README.md`)

**Seeds:** Every production, baseline, and historical comparison rung is run with **three training seeds (42, 43, 44)**. See `docs/SEEDS.md`.

All rungs share validation-MSD **checkpointing**. Legacy vs BAP-MOS still differ on **reward**, **ring geometry**, and **decoder loss** (see `experiments/prostate/legacy/README.md`).

## Main rungs

1. **BAP-MOS (prostate)** — `outer_loop/tpe/` → `inner_loop/` (kervadec-style + composite); companion random / greedy / heuristic search ablations; independent MedSAM TPE at `outer_loop/medsam/` → `inner_loop/medsam/`
2. **BAP-MOS (bladder)** — `outer_loop/{sam,medsam}/` (max_epochs 15 / patience 4) → `inner_loop/{sam,medsam}/` (max_epochs 100 / patience 20; Bladder MSD)
3. Baselines — `experiments/*/baselines/` (`unet`, `nnunet`, `medsam`); SAM-box ≡ `legacy/box` (same entrypoint)

How `--config` vs `--version` applies selected hyperparameters: `docs/INNER_OUTER_LOOP.md`.

## Historical rungs (legacy protocol)

**Prostate** (`experiments/prostate/legacy/`):

1. Box-only — `box/`
2. Point-only — `point/`
3. Fixed box:point ratios — `box_point/`
4. Fixed box / point / both — `boxpoint_box_point/`
5. Policies — `policies/` (`ucb1_global`, `ucb1_per_organ`, `epsilon_decay_per_organ`, `epsilon_greedy_per_organ`)

**Bladder** (`experiments/bladder/legacy/`): `box/` and `point/` only. Bandit policies are **not** shipped for bladder (see `experiments/bladder/legacy/policies/README.md`).

Configs live under `experiments/{prostate,bladder}/...`.
