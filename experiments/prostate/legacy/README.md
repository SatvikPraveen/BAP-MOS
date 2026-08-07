# Legacy prompt / policy ablations (prostate)

> **Historical only.** These configs are earlier MSD-only, fixed-ring, CE+Dice experiments.
> They are **not** part of the final BAP-MOS protocol and are **not required** to reproduce
> the main paper method tables. These runs may be useful for historical comparison or sanity
> checks, but they **do not appear in any main paper tables.** Configs remain **runnable for
> curiosity** but are not recommended for new work. For the final method see `../bapmos/` and
> `docs/INNER_OUTER_LOOP.md`.

## Protocol (legacy)

- Reward: absolute / **negated** validation MSD
- `reward_clip_max_msd_mm=20`
- Fixed `ring_width=20` (not `scale_organ`)
- Loss: **CE + Dice** (not kervadec-style) — enforced in
  `bapmos.legacy.optimization.trainer` via `force_non_method_ce_dice_loss`.
  Fixed-ratio prompting (`box_point`, `boxpoint_box_point`) and policies use
  the same non-method loss as box/point.

## Not the same as BAP-MOS

Final BAP-MOS lives under `experiments/prostate/bapmos/` and uses:

- Composite reward (`composite_fixed_clip`, λ=3)
- `scale_organ` ring
- Kervadec-style boundary loss on the inner-loop (production) path

Do not mix metrics or hyperparameters across legacy rungs and `bapmos/` without noting the protocol change.

## Ladder order inside `legacy/`

1. `box/`
2. `point/`
3. `box_point/` — **10/90, 50/50, 90/10** only (canonical ratio set)
4. `boxpoint_box_point/` — **1:1:1, 1:5:1, 2:3:5, 5:1:1** only (canonical mix set)
5. `policies/*` (`ucb1_global`, `ucb1_per_organ`, `epsilon_decay_per_organ`, `epsilon_greedy_per_organ`)

Library support for these runs lives under `src/bapmos/legacy/` (e.g. historical bandit trainer).

Baselines and the final method stay at the case root: `../baselines/`, `../bapmos/`.

## Running (curiosity only)

Entry points are noted in each `config.yaml`. In short:

- Box / point: `python -m bapmos.multiorgan.train_sam_multiorgan_decoder_{box,points}`
- Ratios / policies: `python -m bapmos.legacy.optimization.trainer --config <config.yaml> --experiment <name>`

Run roots: `runs/prostate/legacy/…` (see `docs/RUNTIME_LAYOUT.md`). Test inference and results collation: `docs/RUNNING.md`.

Use the three-seed protocol if reporting (`docs/SEEDS.md`). Prefer `../bapmos/` for any new work.
