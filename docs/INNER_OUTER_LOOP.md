# Outer loop and inner loop

| Loop | Role |
|------|------|
| **outer_loop** | Optuna HPO / search on validation MSD (TPE; ablations: random, greedy, heuristic) |
| **inner_loop** | Full production train with **selected** hyperparameters (kervadec-style + composite reward) |

## Protocol defaults (final BAP-MOS)

| Piece | Setting |
|-------|---------|
| Loss | **BAP-MOS method** (SAM or MedSAM init): Kervadec-style (`loss_mode: kervadec`). **All other paths** (UNet, MedSAM+box baseline, SAM+box/points, fixed-ratio prompting, legacy policies, nnU-Net): regional CE+Dice / nnU-Net Dice+CE — never Kervadec |
| Reward | Composite (`composite_fixed_clip`, λ=3) |
| Checkpoint | Validation MSD (`ptv_kfold_msd`; Bladder organ on PFUS1) |
| HPO objective | Same validation MSD |
| Loops | **Outer loop** = TPE search (+ ablations); **Inner loop** = production train |
| Training seeds | Three seeds (42, 43, 44), mapped to experiment run names `pooled_seed42`, `pooled_seed43_rep2`, `pooled_seed44_rep3` |

Encoded in `configs/common/protocol.yaml` and merged as the compose floor
(`compose_config`). Suite YAMLs override. See `docs/SEEDS.md` for seed naming.

### Loss policy

| Path | Loss |
|------|------|
| **BAP-MOS method** (`bapmos.method`, Meta SAM or MedSAM init) | Kervadec-style (`training.loss_mode: kervadec`) — enforced in the trainer |
| **External baselines** (UNet, MedSAM-init+box, nnU-Net) | Regional CE+Dice (nnU-Net: stock Dice+CE) |
| **Multiorgan SAM box / points** (paper baselines, not BAP-MOS) | Regional CE+Dice |
| **Legacy fixed-ratio prompting** (`box` / `point` / `box_point` / `boxpoint_box_point`) and **legacy policies** | Regional CE+Dice (enforced in `legacy.optimization.trainer`) |
| **Legacy** historical notes | Same CE+Dice protocol (not Kervadec) |

See `bapmos.losses.loss_policy`.

### Checkpoints

- **outer_loop (Optuna):** After each trial’s validation MSD is written to Optuna, trial `*.pth` under `runs/.../outer_loop/.../` are **deleted** (search only needs the score). Mid-train failures keep `last_checkpoint.pth` so jobs can resume.
- **outer_loop (trainer CLI):** When `evaluation.cleanup_checkpoints_after_train: true`, the trainer deletes `*.pth` in the run dir **after a successful train**.
- **inner_loop:** Production checkpoints are **kept** (`cleanup_checkpoints_after_train: false`).

### Train-time test vs stratified inference_output

`run_test_after_train: true` writes **metrics/JSON only** under the run dir. It does **not** implement the stratified PNG/PDF `inference_output/` protocol.

After production train, export panels with:

```bash
python -m bapmos.method.run_test_inference --help
```

Default export is seed-42 only; for three-seed mean ± std, export seeds 43/44 with
`--force-inference-output` into per-seed `inference_output/` dirs (see `docs/SEEDS.md` / `docs/RUNNING.md`).

### Selected hyperparameters

Committed `inner_loop/**/selected/*.yaml` files may start as **`generation: 0` placeholders**.

1. Finish **outer_loop** search.
2. **Export** best trial so it **overwrites** the matching `selected/*.yaml`.
3. Only then start **inner_loop** production (`--version bapmos …` / `bapmos_sam` / …).

Do **not** treat placeholder `selected/` as paper hyperparameters.

## Prostate

- Meta SAM search: `experiments/prostate/bapmos/outer_loop/{tpe,random,greedy,heuristic}/version.yaml`
  - Compose keys: `bapmos_outer_loop` (TPE), `bapmos_outer_loop_random`, `bapmos_outer_loop_greedy`, `bapmos_outer_loop_heuristic`
  - HPO suites: `bapmos_bo`, `bapmos_bo_random`, `bapmos_bo_greedy`, `bapmos_bo_heuristic`
  - Exports: `inner_loop/selected/<method>/pooled.yaml`
  - Production: `--version bapmos --dataset pooled`
- MedSAM search (independent TPE): `experiments/prostate/bapmos/outer_loop/medsam/version.yaml`
  - Compose: `bapmos_medsam_pooled_outer_loop`; HPO suite: `bapmos_bo_medsam_pooled`
  - Same per-trial budget as Meta SAM outer (**100** epochs / patience **15**); MedSAM weights
  - Export: `inner_loop/medsam/selected/pooled.yaml`
  - Production: `--version bapmos_medsam_pooled --dataset pooled`

## Bladder (PFUS1)

- Search: `experiments/bladder/bapmos/outer_loop/{sam,medsam}/version.yaml`
  (`bapmos_bo_sam` / `bapmos_bo_medsam`; compose `bapmos_sam_outer_loop` / `bapmos_medsam_outer_loop`)
- Production SAM: `--version bapmos_sam --dataset pfus1`
- Production MedSAM: `--version bapmos_medsam --dataset pfus1`
  (PFUS1 production: max_epochs **100**, patience **20**; outer search: max_epochs **15**, patience **4**)
- Bladder outer bandit is a **cheaper proxy** of inner (warmup **5** / min_pulls **3** vs inner **10** / **5**);
  `val_msd_min_delta` is **0.0** on both (same improvement definition; only train length differs).

## How to run (strict)

**Production and paper reproduction must use compose:**

```bash
python -m bapmos.method.bap_mos_trainer \
  --version bapmos --dataset pooled --experiment pooled_seed42
```

| CLI | Behavior |
|-----|----------|
| `--version <suite> --dataset <name>` | **Required for production.** Merges site `common.yaml` + suite `version.yaml` + `selected/` when `ablation.parameter_source: bayesian_selected`. |
| `--config <yaml>` | Loads YAML **as-is**. Does **not** merge site common or `selected/`. Prefer `--version` + `--dataset` for paper reproduction. |

Suite defaults for `selected/` merge:

| Suite | Default selected folder |
|-------|-------------------------|
| `bapmos` | `selected/tpe` |
| `bapmos_sam` / `bapmos_medsam` | `selected` (under that backbone’s inner_loop dir) |
| `bapmos_medsam_pooled` | `selected` (under `inner_loop/medsam/`; from `bapmos_bo_medsam_pooled` export) |

To train a prostate search ablation other than TPE, set `BAPMOS_SELECTED_SUBDIR=selected/<method>`
(or the bare name `random` / `greedy` / `heuristic`). Prostate SAM production run roots are
isolated per method: `runs/prostate/bapmos/inner_loop/{tpe,random,greedy,heuristic}/`.

Placeholder `selected/` files (`selection_meta.generation: 0`) are refused at compose time
until outer-loop export overwrites them.

```bash
# Prostate main (TPE export)
python -m bapmos.method.bap_mos_trainer \
  --version bapmos --dataset pooled --experiment pooled_seed42

# Prostate random-search export
BAPMOS_SELECTED_SUBDIR=selected/random python -m bapmos.method.bap_mos_trainer \
  --version bapmos --dataset pooled --experiment pooled_seed42

# Bladder MedSAM
python -m bapmos.method.bap_mos_trainer \
  --version bapmos_medsam --dataset pfus1 --experiment pfus1_seed42

# Prostate MedSAM (after outer-loop export from bapmos_bo_medsam_pooled)
python -m bapmos.method.bap_mos_trainer \
  --version bapmos_medsam_pooled --dataset pooled --experiment pooled_seed42
```

## Legacy policy experiments

Before the final BAP-MOS design, we compared prompt mixes and bandit policies (UCB1, ε-greedy, ε-decay) under an **older** protocol:

- Reward: negated validation MSD only (not composite)
- Ring: fixed `ring_width` (not `scale_organ`)
- Loss: CE + Dice (not kervadec-style)

Those runs live under `experiments/*/legacy/` and use historical trainers under `src/bapmos/legacy/`. They document the progression to BAP-MOS and are **not required** to reproduce the main paper method tables. See each `legacy/README.md`.
