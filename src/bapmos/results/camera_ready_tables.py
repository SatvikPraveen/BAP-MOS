"""Collate camera-ready exports into mean ± std tables (uses :mod:`bapmos.results.collate_seeds`).

Walks ``inference_output/camera_ready/<group>/<run_name>/`` and ingests every finished export
(``sites.json`` for pooled TRUS, ``metrics/summary_metrics.csv`` for PFUS1) under a method id
derived from its folder, then builds the same tables as the paper's collation: slice-weighted
pooling across TRUS sites, then mean ± std over seeds 42/43/44.

Method ids: ``<group>`` for single-method groups (``unet``, ``medsam``, ``nnunet`` …, plus
the dataset tag in the run name, e.g. ``unet_pooled``), and ``<group>_<variant>`` otherwise
(``suite_global``, ``noise10_reference``, ``table1_multi_box_full``).

    python -m bapmos.results.camera_ready_tables --out <dir outside the repo>

Output goes to ``<out>/results/<corpus>/...`` so the repository's ``results/`` stays as is.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from bapmos.results.collate_seeds import build_corpus, ingest_inference_run

SEED_SUFFIX = re.compile(r"_(seed42|seed43_rep2|seed44_rep3)$")
SEED_OF = {"seed42": 42, "seed43_rep2": 43, "seed44_rep3": 44}


def finished_exports(root: Path) -> Iterator[Tuple[str, str, str, int, Path]]:
    """Yield ``(corpus, method, run_name, seed, run_dir)`` for every finished export."""
    for group_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            m = SEED_SUFFIX.search(run_dir.name)
            if not m:
                continue
            pooled = (run_dir / "sites.json").is_file()
            flat = (run_dir / "metrics" / "summary_metrics.csv").is_file()
            if not (pooled or flat):
                continue
            stem = run_dir.name[: m.start()]
            corpus = "bladder" if "pfus1" in run_dir.name or "pfus1" in group_dir.name else "prostate"
            method = f"{group_dir.name}_{stem}" if stem else group_dir.name
            if group_dir.name in ("unet", "medsam", "box", "nnunet"):
                method = stem  # e.g. unet_pooled, unet_loco_case1, medsam_pfus1
            yield corpus, method, run_dir.name, SEED_OF[m.group(1)], run_dir


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--exports", default="inference_output/camera_ready")
    p.add_argument("--out", required=True, help="Tables root (outside the repository)")
    args = p.parse_args(argv)
    out = Path(args.out).resolve()
    corpora = set()
    n = 0
    for corpus, method, run_name, seed, run_dir in finished_exports(Path(args.exports)):
        dataset = "pfus1" if corpus == "bladder" else "pooled"
        ingest_inference_run(
            run_dir, corpus=corpus, method=method, run_name=run_name,
            dataset=dataset, seed=seed, repo=out,
        )
        corpora.add((corpus, dataset))
        n += 1
    for corpus, dataset in sorted(corpora):
        build_corpus(corpus, dataset=dataset, repo=out)
    print(f"ingested {n} exports; tables under {out}/results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
