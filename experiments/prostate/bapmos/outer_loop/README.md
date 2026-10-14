# BAP-MOS outer loop (prostate)

Search over the BAP-MOS hyperparameters: reward clip τ, ring scale α and floor r_min, bandit
window W and block size K. The objective is fold-balanced validation MSD on the PTV, the same
metric used for checkpointing.

| Method | Suite (`--hpo-suite`) | Compose version | Config |
|--------|-----------------------|-----------------|--------|
| TPE, SAM (main) | `bapmos_bo` | `bapmos_outer_loop` | `tpe/version.yaml` |
| Random, SAM | `bapmos_bo_random` | `bapmos_outer_loop_random` | `random/version.yaml` |
| Greedy, SAM | `bapmos_bo_greedy` | `bapmos_outer_loop_greedy` | `greedy/version.yaml` |
| Heuristic, SAM | `bapmos_bo_heuristic` | `bapmos_outer_loop_heuristic` | `heuristic/version.yaml` |
| TPE, MedSAM | `bapmos_bo_medsam_pooled` | `bapmos_medsam_pooled_outer_loop` | `medsam/version.yaml` |

Each trial trains for at most 100 epochs (patience 15) with the same loss as the inner loop.
Every method has a budget of 100 trials ([`docs/SEARCH_METHODS.md`](../../../../docs/SEARCH_METHODS.md)).
The MedSAM search is a separate study with its own export.

```text
outer_loop/
  {tpe,random,greedy,heuristic}/version.yaml   # SAM
  medsam/version.yaml                          # MedSAM
inner_loop/
  version.yaml                                 # BAP-MOS (SAM)
  selected/{tpe,random,greedy,heuristic}/pooled.yaml
  medsam/version.yaml                          # BAP-MOS (MedSAM)
  medsam/selected/pooled.yaml
```

Run a search with `init`, then `run`:

```bash
python -m bapmos.hpo.study_runner init --hpo-suite bapmos_bo --dataset pooled
HPO_SEQUENTIAL=1 python -m bapmos.hpo.study_runner run \
  --hpo-suite bapmos_bo --dataset pooled --n-trials 100
```

The committed `selected/` files already hold the paper's exported values, so the search is
optional. `study_runner export` overwrites them with your study's best trial.

Inner-loop training: [`../inner_loop/README.md`](../inner_loop/README.md). Details:
[`docs/INNER_OUTER_LOOP.md`](../../../../docs/INNER_OUTER_LOOP.md).
