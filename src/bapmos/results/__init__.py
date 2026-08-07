"""Paper tables: per-seed metrics + mean±std across seeds 42/43/44.

Layout (gitignored runtime tree; see ``results/README.md``)::

    results/
      prostate/pooled/
        by_seed/<method>/<run_name>.csv
        combined/prostate_pooled_mean_std.csv
        combined/prostate_pooled_ptv_mean_std.csv
      bladder/pfus1/
        by_seed/<method>/<run_name>.csv
        combined/bladder_pfus1_mean_std.csv
        combined/bladder_pfus1_bladder_mean_std.csv
"""

from bapmos.results.layout import (
    PROSTATE_METHOD_IDS,
    BLADDER_METHOD_IDS,
    corpus_results_root,
    method_by_seed_dir,
    method_combined_dir,
)

__all__ = [
    "PROSTATE_METHOD_IDS",
    "BLADDER_METHOD_IDS",
    "corpus_results_root",
    "method_by_seed_dir",
    "method_combined_dir",
]
