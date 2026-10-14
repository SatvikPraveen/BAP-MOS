# Outer-loop search methods

The outer loop searches five BAP-MOS hyperparameters (the `searched_clip_scale_organ` space,
defined in `src/bapmos/hpo/search_space.py`):

| Key | Role | Range |
|-----|------|-------|
| `clip_max_mm` | Reward clip τ | integer in [5, 30] |
| `alpha` | Ring scale α | [0.05, 0.15] |
| `r_min` | Ring floor r_min (px) | integer in [2, 20] |
| `window_size` | Sliding reward window W | {20, 25, …, 100} |
| `block_size_batches` | Decision block K (batches) | {20, 25, …, 100} |

The objective is fold-balanced (5 folds) validation MSD on the primary organ, minimised. Every
method has a budget of 100 trials, so the methods are directly comparable.

The search-strategy comparison is defined for prostate only, under
`experiments/prostate/bapmos/outer_loop/{tpe,random,greedy,heuristic}/`. PFUS1 uses TPE with the
same search space, once per backbone (`bapmos_bo_sam`, `bapmos_bo_medsam`).

## Methods

| Method | Suite | How it proposes trials |
|--------|-------|------------------------|
| TPE (main) | `bapmos_bo` | 20 start-up trials (1 baseline + 19 Latin-hypercube samples, seed 42), then Optuna TPE |
| Random | `bapmos_bo_random` | Optuna `RandomSampler` over the same space |
| Greedy | `bapmos_bo_greedy` | Coordinate-wise search over a 7-value grid per dimension |
| Heuristic | `bapmos_bo_heuristic` | Fixed, deterministic catalog of candidates |

- TPE adapts its proposals to previous results.
- Random samples independently of previous results.
- Greedy is sequential: later probes start from the best configuration found so far.
- Heuristic evaluates a fixed list; nothing adapts.

The MedSAM search for prostate (`bapmos_bo_medsam_pooled`,
`experiments/prostate/bapmos/outer_loop/medsam/`) is a separate TPE study with the same space
and budget.

Configs: `experiments/prostate/bapmos/outer_loop/<method>/version.yaml`. Exports:
`experiments/prostate/bapmos/inner_loop/selected/<method>/pooled.yaml` and
`experiments/prostate/bapmos/inner_loop/medsam/selected/pooled.yaml`.

Every method runs as `study_runner init`, then `study_runner run`
([`RUNNING.md`](RUNNING.md#outer-loop-optional)).

## Greedy

1. `init` enqueues the baseline trial (`clip=20`, `alpha=0.1`, `r_min=6`, `window=50`,
   `block=50`) and writes `greedy_state.json` next to the study database.
2. `run` trains it, then works through the dimensions in order `clip_max_mm`, `alpha`, `r_min`,
   `window_size`, `block_size_batches`. For each dimension it probes 7 grid values with the
   other four at the current best, keeps the best, and moves to the next dimension.
3. Sweeps repeat until the budget of 100 trials is used.

`run` finalizes each wave and enqueues the next one itself. The subcommands
`greedy-baseline-finalize`, `greedy-wave --wave <0-4>` and `greedy-finalize --wave <0-4>` do the
same steps by hand, for example when several workers share one study.

## Heuristic

Every trial is listed in advance; Optuna only runs the queue (`sampler: catalog`). `init`
enqueues the whole catalog. The catalog (`heuristic_catalog()` in
`src/bapmos/hpo/catalog_search.py`) is built in tiers, deduplicated, and capped at 100 trials:

1. Baseline (`clip=20`, `alpha=0.1`, `r_min=6`, `window=50`, `block=50`).
2. One-dimension sweeps over clip, alpha, r_min, window and block, others at baseline.
3. A sparse geometry lattice: `clip ∈ {5, 13, 21, 30}` × `alpha ∈ {0.05, 0.10, 0.15}` ×
   `r_min ∈ {2, 8, 14, 20}`, bandit settings at baseline.
4. A bandit grid: `window × block` on `{20, 40, 60, 80, 100}`, geometry at baseline.

Clip, alpha and r_min levels reuse the greedy 7-value grids.

## Related files

| File | Contents |
|------|----------|
| `src/bapmos/hpo/paths.py` | Suite definitions and study folders |
| `src/bapmos/hpo/study_runner.py` | `init`, `run`, `export` and the greedy subcommands |
| `src/bapmos/hpo/startup_catalog.py` | TPE start-up catalog |
| `src/bapmos/hpo/catalog_search.py` | Greedy grids and the heuristic catalog |
| `src/bapmos/hpo/search_space.py` | Search space |
| `src/bapmos/hpo/saturation_stop.py` | Early stop for TPE and random studies |
