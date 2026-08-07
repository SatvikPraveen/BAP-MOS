# Outer-loop search methods

Methods currently defined for **prostate** under `experiments/prostate/bapmos/outer_loop/{tpe,random,greedy,heuristic}/`.

**Bladder** does **not** ship that search-method family. It uses backbone suites under `experiments/bladder/bapmos/outer_loop/{sam,medsam}/` (`bapmos_bo_sam` / `bapmos_bo_medsam`) with the same 5-D `searched_clip_scale_organ` box and TPE. See `docs/EXPERIMENT_LADDER.md` and `experiments/bladder/bapmos/outer_loop/README.md`.

BAP-MOS searches five hyperparameters in the `searched_clip_scale_organ` box:

| Key | Role |
|-----|------|
| `clip_max_mm` | Composite-reward clip τ |
| `alpha` | Organ-scale ring scale |
| `r_min` | Organ-scale ring floor (px) |
| `window_size` | Sliding bandit memory |
| `block_size_batches` | Decision-block length |

**Objective:** validation PTV k-fold MSD (minimize). Default budget: **100 trials** for all methods.

All methods use the same objective (validation MSD) and nominal trial budget (100), making search-method comparisons directly comparable.

## Methods in the public BAPMOS tree

| Method | Included? | How it proposes trials |
|--------|-----------|-------------------------|
| **TPE** | Yes (main) | Optuna TPE after a startup catalog |
| **Random** | Yes (ablation) | Optuna `RandomSampler` |
| **Greedy** | Yes (ablation) | Sequential coordinate-wise search over a fixed 7-point grid per dim |
| **Heuristic** | Yes (ablation) | Fixed multi-tier catalog (no adaptive sampler) |

**TPE** adaptively proposes trials based on previous observations.  
**Random** search samples independently from the same search space (stochastic, but not adaptive to observed objectives).  
**Greedy** performs sequential coordinate-wise search, where later probes depend on the best configurations found in earlier waves.  
**Heuristic** evaluates a fixed, deterministic catalog of candidates.

Configs: `experiments/prostate/bapmos/outer_loop/{tpe,random,greedy,heuristic}/version.yaml`
(plus independent MedSAM TPE at `outer_loop/medsam/version.yaml`).  
Compose versions: `bapmos_outer_loop` / `_random` / `_greedy` / `_heuristic` / `bapmos_medsam_pooled_outer_loop`.  
Selected exports: `inner_loop/selected/{tpe,random,greedy,heuristic}/pooled.yaml` and
`inner_loop/medsam/selected/pooled.yaml`.

---

## Heuristic search

Heuristic is a **deterministic fixed catalog**: every trial is pre-listed; Optuna only executes the queue (`sampler: catalog`). There is no Bayesian / random adaptive proposal.

Experiment path: `experiments/prostate/bapmos/outer_loop/heuristic/`  
Export: `experiments/prostate/bapmos/inner_loop/selected/heuristic/pooled.yaml`

### Algorithm (`heuristic_catalog`)

Implemented in `src/bapmos/hpo/catalog_search.py` (`heuristic_catalog()`); see related helpers below and inline docstrings for suite / path details.

Candidates are built in **priority tiers**, deduplicated, then capped at `max_trials` (default 100):

1. **Baseline** — enqueue point (`clip=20`, `alpha=0.1`, `r_min=6`, `window=50`, `block=50`)
2. **Univariate sweeps** — vary one of clip / alpha / r_min / window / block over coarse grids; others fixed at baseline
3. **Sparse 3-D geometry lattice** — `clip ∈ {5,13,21,30}` × `alpha ∈ {0.05,0.10,0.15}` × `r_min ∈ {2,8,14,20}`; bandit dims at baseline
4. **Bandit joint grid** — `window × block` on `{20,40,60,80,100}`; geometry at baseline

Grid levels for clip/alpha/r_min reuse the same 7-point greedy grids; window/block use 5-point `{20…100}`.

### How it is run

- Suite id: `bapmos_bo_heuristic`
- Spec in `src/bapmos/hpo/paths.py`:
  - `search_method: heuristic`
  - `sampler: catalog`
  - `selected_subdir: selected/heuristic`
  - Study and trial subdir names: see `src/bapmos/hpo/paths.py` (outer_loop search / inner_loop production naming)
- Enqueue: `study_runner.enqueue_heuristic_catalog()` walks `heuristic_catalog()` and adds trials
- Saturation stop treats heuristic like greedy (catalog methods) in `src/bapmos/hpo/saturation_stop.py`

### Related files (heuristic)

| File | Role |
|------|------|
| `src/bapmos/hpo/catalog_search.py` | `heuristic_catalog`, grids, greedy helpers |
| `src/bapmos/hpo/paths.py` | Suite `bapmos_bo_heuristic` and subdir names |
| `src/bapmos/hpo/study_runner.py` | `enqueue_heuristic_catalog`, study create/run |
| `src/bapmos/hpo/search_space.py` | Maps suite → `searched_clip_scale_organ` |
| `experiments/prostate/bapmos/inner_loop/selected/heuristic/pooled.yaml` | Exported (or placeholder) best params |

---

## Greedy (brief)

- Baseline trial, then waves: for each dim in `(clip, alpha, r_min, window, block)`, probe 7 grid values with others fixed at current best; pick best; next dim; repeat until 100 trials.
- Later waves depend on earlier best configurations (sequential coordinate-wise search).
- State: `greedy_state.json` under the study directory.
- CLI helpers in `src/bapmos/hpo/study_runner.py`: `greedy-wave`, `greedy-finalize`, `greedy-baseline-finalize`.

## Random (brief)

- Optuna `RandomSampler` over the same continuous/discrete search space as TPE; same trial budget.
- Samples independently of previous objective values (stochastic, non-adaptive).
