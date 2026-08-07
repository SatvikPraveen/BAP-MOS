# Training seeds (three-seed protocol)

All **main-path production runs**, **baselines**, and **historical comparison runs intended for reporting** use **three independent training seeds**.

## Canonical names (use these)

Experiment name **is** the run folder name (no separate `--run-name` needed):

| Replicate | Training seed | Prostate (`run_root` under `runs/prostate/...`) | Bladder |
|-----------|---------------|--------------------------------------------------|---------|
| primary   | **42**        | `pooled_seed42`                                  | `pfus1_seed42` |
| rep2      | **43**        | `pooled_seed43_rep2`                             | `pfus1_seed43_rep2` |
| rep3      | **44**        | `pooled_seed44_rep3`                             | `pfus1_seed44_rep3` |

Helpers: `bapmos.replicate_runs` (`replicate_training_seed`, `replicate_run_name`, `all_study_run_names`).

```python
from bapmos.replicate_runs import all_study_run_names

all_study_run_names("pooled_seed42")
# -> ("pooled_seed42", "pooled_seed43_rep2", "pooled_seed44_rep3")
```

Layout:

```text
runs/prostate/bapmos/outer_loop/tpe/<trial>/
runs/prostate/bapmos/inner_loop/tpe/pooled_seed42/       # also random|greedy|heuristic/
runs/prostate/bapmos/inner_loop/medsam/pooled_seed42/
runs/bladder/bapmos/inner_loop/sam/pfus1_seed42/
inference_output/prostate/pooled/<method>/<run_name>/   # per-seed stratified export
inference_output/bladder/pfus1/<method>/<run_name>/
results/prostate/pooled/by_seed/<method>/<run>.csv      # per-seed metrics (all seeds)
results/bladder/pfus1/…
```

Mean ± std tables: ingest each seed’s export, then
`python -m bapmos.results.collate_seeds build --corpus prostate` (see `docs/RESULTS.md`).

## Rules

- **Outer loop (search):** training seed **42** only; find HPs, then export `selected/`.
- **Inner loop (production):** start **only after** outer-loop export **overwrites** `selected/` placeholders (`generation: 0`); train with seeds **42, 43, 44**.
- Report mean ± std (or all three) over seeds **42, 43, 44**.
- Keep **checkpoint k-fold seed** (`evaluation.kfold_seed`) and **probe seed** fixed at **42** across all three training seeds (vary only the training seed).
- Each seed gets an **isolated** folder under `runs/.../<run_name>/`.
- Inner-loop YAML sets `evaluation.run_test_after_train: false`; run stratified export **after** training (see below).
- Pooled prostate test lists live under `data/prostate/pooled/site_tests/<site>/test.txt` (no global `test.txt`).
- Prostate SAM productions for different search methods write under
  `runs/prostate/bapmos/inner_loop/<method>/` (no shared `pooled_seed*` collision).
- Compose refuses `generation: 0` placeholder `selected/` until outer-loop export overwrites them.

## Stratified inference export (canonical three-seed path)

By default, `run_test_inference` writes stratified `inference_output/` for the **primary seed-42**
run only (prediction + matplotlib PNG/PDF for the first 10 evaluated slices @ 350 dpi; metrics
on the full test split).

To reproduce **three-seed mean ± std** tables, export **all three** seeds into per-seed
directories under `inference_output/`, then collate from those dirs:

1. Seed 42 may omit `--force-inference-output`.
2. Seeds 43/44 **require** `--force-inference-output`.
3. Prefer a uniform layout: `inference_output/.../<method>/<run_name>/` for every seed.
4. Ingest each seed from that path, then `collate_seeds build` (see `docs/RUNNING.md`).

Do **not** expect seed-43/44 metrics under `runs/.../test_metrics/` for the main protocol
(`run_test_after_train: false`).

## Do not

- Treat a single seed-42 run as the paper result.
- Reuse the same run directory across seeds.
- Ingest seeds 43/44 from `inference_output/` without having exported them with `--force-inference-output`.
- Start inner loop on `generation: 0` placeholder `selected/` files — wait for outer-loop export.
