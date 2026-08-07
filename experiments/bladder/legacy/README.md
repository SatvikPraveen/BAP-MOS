# Legacy prompt ablations (bladder PFUS1)

> **Historical only.** Earlier prompt comparisons — **not** the final BAP-MOS protocol
> and **not required** for main paper method tables. These runs may be useful for
> historical comparison, but they **do not appear in any main paper tables.**
> For the final method see `../bapmos/` and `docs/INNER_OUTER_LOOP.md`.

## What is shipped here

1. `box/` — box-only prompt baseline
2. `point/` — point-only prompt baseline

Full protocol note (negated MSD reward, fixed ring, CE+Dice — never Kervadec):
**`experiments/prostate/legacy/README.md`**. Loss is enforced in
`bapmos.legacy.optimization.trainer` (non-method path).

## What is not shipped

- Bandit policies (UCB1 / epsilon) were **not** ported to bladder in this public tree.
  See `policies/README.md` and `experiments/prostate/legacy/policies/`.

Final method and baselines remain at the case root: `../bapmos/`, `../baselines/`.
