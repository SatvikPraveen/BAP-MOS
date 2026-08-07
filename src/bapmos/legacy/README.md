# Legacy library modules

Historical trainers and helpers for prompt/policy ablations under
`experiments/*/legacy/`. **Not required** to reproduce main BAP-MOS results
(`experiments/*/bapmos/` + `experiments/*/baselines/`).

Prefer imports via `bapmos.legacy.*`. Root-level package shims (`bapmos.optimization`,
`singleorgan`, `ablations`, `pfus1_advanced`) alias the same modules via
`bapmos._legacy_aliases`.

W&B entity (legacy trainers / single-organ) comes from `--wandb_entity` or
`WANDB_ENTITY` only — never a hardcoded org.

See `experiments/prostate/legacy/README.md` for the legacy protocol note.
