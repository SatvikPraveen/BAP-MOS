"""Recompute Dice / MSD / HD95 on the native GT grid with physical spacing.

Primary paper protocol (feedback-aligned):
  - Nearest-neighbor align pred → GT shape if needed
  - Dice on that grid
  - MSD / HD95 with taxonomy ``pixel_spacing_mm`` (true mm)
  - Do **not** overwrite existing export ``metrics/``; write ``metrics_native_redone/``
    under inference runs, then collate into published ``by_seed/`` + ``combined/*_mean_std.csv``

Examples::

  python -m bapmos.results.recompute_native_metrics audit --corpus prostate
  python -m bapmos.results.recompute_native_metrics run --corpus prostate
  python -m bapmos.results.recompute_native_metrics run --corpus bladder
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from bapmos.paths import find_combined_masks_dir, project_root
from bapmos.results.collate_seeds import (
    build_mean_std_rows,
    pool_organs_from_per_slice,
    write_normalized_csv,
)
from bapmos.results.layout import (
    DEFAULT_DATASET,
    POOLED_SITE_LABEL,
    PROSTATE_TEST_SITES,
    corpus_results_root,
    experiment_key_from_run_name,
    infer_seed_from_run_name,
)
from bapmos.legacy.optimization.metrics import MetricsEvaluator
from bapmos.train.training_taxonomy import get_baseline_taxonomy_profile

logger = logging.getLogger(__name__)

_SEED_RUN_RE = re.compile(r"seed(\d+)")


@dataclass(frozen=True)
class RunSpec:
    method: str
    run_name: str
    run_dir: Path
    seed: int


def _canonical_run_name(method: str, run_name: str) -> str:
    """Match ``by_seed`` keys used by collate.

    Examples: ``unet_pooled_seed42`` → ``pooled_seed42``;
    ``unet_pfus1_seed42_native_redone`` → ``pfus1_seed42``.
    Leaves families like ``ucb1_global_seed42`` intact.
    """
    name = str(run_name).strip()
    if name.endswith("_native_redone"):
        name = name[: -len("_native_redone")]
    prefix = f"{method}_"
    if name.startswith(prefix):
        rest = name[len(prefix) :]
        if rest.startswith("pooled") or rest.startswith("pfus1"):
            return rest
    return name


def _inference_root(corpus: str, repo: Path) -> Path:
    if corpus == "prostate":
        return repo / "inference_output" / "prostate" / "pooled"
    if corpus == "bladder":
        return repo / "inference_output" / "bladder" / "pfus1"
    raise ValueError(corpus)


def _data_root(corpus: str, repo: Path) -> Path:
    if corpus == "prostate":
        return repo / "data" / "prostate" / "pooled"
    if corpus == "bladder":
        return repo / "data" / "bladder" / "pfus1"
    raise ValueError(corpus)


def _stem_from_split_token(token: str) -> str:
    """Map split tokens to mask/pred stems.

    Examples: ``slice_001.png`` → ``slice_001``;
    ``P001/frame_000`` → ``P001_frame_000`` (PFUS1).
    """
    token = token.strip().replace("\\", "/")
    if "/" in token:
        parts = [Path(p).stem for p in token.split("/") if p]
        return "_".join(parts)
    return Path(token).stem


def _load_test_stems(data_root: Path, *, site: Optional[str] = None) -> List[str]:
    if site is None:
        # bladder single split
        for cand in (
            data_root / "splits_patient_70_15_15_seed42" / "test.txt",
            data_root / "splits_stratified" / "test.txt",
        ):
            if cand.is_file():
                return [
                    _stem_from_split_token(line.strip().split()[0])
                    for line in cand.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
        raise FileNotFoundError(f"No test split under {data_root}")
    test_txt = data_root / "site_tests" / site / "test.txt"
    if not test_txt.is_file():
        raise FileNotFoundError(test_txt)
    stems: List[str] = []
    for line in test_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        stems.append(_stem_from_split_token(line.split()[0]))
    return stems


def _discover_runs(corpus: str, repo: Path) -> List[RunSpec]:
    root = _inference_root(corpus, repo)
    if not root.is_dir():
        return []
    out: List[RunSpec] = []
    for method_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        method = method_dir.name
        if method.startswith("_"):
            continue
        for run_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            run_name = run_dir.name
            # skip non-seed clutter (e.g. nnunet site folders at family root)
            if corpus == "prostate" and run_name in PROSTATE_TEST_SITES:
                continue
            if run_name in ("metrics", "predictions", "visualizations"):
                continue
            seed = infer_seed_from_run_name(run_name)
            if seed is None:
                m = _SEED_RUN_RE.search(run_name)
                seed = int(m.group(1)) if m else None
            if seed is None:
                continue
            # must look like an inference run
            if corpus == "prostate":
                if not any((run_dir / s / "predictions").is_dir() for s in PROSTATE_TEST_SITES):
                    continue
            else:
                if not (run_dir / "predictions").is_dir() and not (
                    run_dir / "metrics"
                ).is_dir():
                    continue
            out.append(
                RunSpec(method=method, run_name=run_name, run_dir=run_dir, seed=int(seed))
            )
    return out


def _pred_ids_path(run_dir: Path, stem: str, *, site: Optional[str] = None) -> Path:
    base = run_dir / site if site else run_dir
    return base / "predictions" / "multiclass" / f"{stem}_pred_ids.png"


def _gt_path(masks_dir: Path, stem: str) -> Path:
    return masks_dir / f"{stem}_combined_mask.png"


def _read_label_png(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(path)
    if arr.ndim == 3:
        raise ValueError(f"expected single-channel class-ID PNG, got shape {arr.shape}: {path}")
    return arr.astype(np.uint8, copy=False)


def audit_corpus(corpus: str, *, repo: Optional[Path] = None) -> Path:
    repo = repo or project_root()
    data_root = _data_root(corpus, repo)
    masks_dir = find_combined_masks_dir(data_root)
    rows: List[Dict[str, Any]] = []

    for spec in _discover_runs(corpus, repo):
        if corpus == "prostate":
            site_iter: Sequence[Optional[str]] = list(PROSTATE_TEST_SITES)
        else:
            site_iter = [None]
        for site in site_iter:
            expected = _load_test_stems(data_root, site=site)
            saved = 0
            missing: List[str] = []
            shape_mismatch = 0
            bad_dtype = 0
            for stem in expected:
                pp = _pred_ids_path(spec.run_dir, stem, site=site)
                gp = _gt_path(masks_dir, stem)
                if not pp.is_file():
                    missing.append(stem)
                    continue
                saved += 1
                try:
                    pred = _read_label_png(pp)
                except Exception:  # noqa: BLE001
                    bad_dtype += 1
                    continue
                if gp.is_file():
                    gt = _read_label_png(gp)
                    if pred.shape != gt.shape:
                        shape_mismatch += 1
            complete = (
                len(missing) == 0
                and shape_mismatch == 0
                and bad_dtype == 0
                and saved == len(expected)
            )
            rows.append(
                {
                    "corpus": corpus,
                    "method": spec.method,
                    "run_name": spec.run_name,
                    "seed": spec.seed,
                    "site": site or "test",
                    "n_expected": len(expected),
                    "n_saved_pred_ids": saved,
                    "n_missing": len(missing),
                    "n_shape_mismatch": shape_mismatch,
                    "n_bad_label_png": bad_dtype,
                    "complete_for_native_recompute": int(complete),
                    "missing_sample": ",".join(missing[:5]),
                }
            )

    out_dir = corpus_results_root(corpus, DEFAULT_DATASET[corpus], repo=repo) / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{corpus}_{DEFAULT_DATASET[corpus]}_pred_audit_redone.csv"
    fields = list(rows[0].keys()) if rows else [
        "corpus",
        "method",
        "run_name",
        "seed",
        "site",
        "n_expected",
        "n_saved_pred_ids",
        "n_missing",
        "complete_for_native_recompute",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info("Wrote audit %s (%d rows)", out_path, len(rows))
    return out_path


def _evaluate_stems(
    *,
    stems: Sequence[str],
    run_dir: Path,
    masks_dir: Path,
    taxonomy,
    site: Optional[str] = None,
) -> MetricsEvaluator:
    class_mapping = {int(k): v for k, v in taxonomy.multiclass_eval_mapping.items()}
    evaluator = MetricsEvaluator(
        pixel_spacing=tuple(taxonomy.pixel_spacing_mm),
        organs=list(taxonomy.evaluator_organ_labels),
    )
    for stem in stems:
        pred = _read_label_png(_pred_ids_path(run_dir, stem, site=site))
        gt = _read_label_png(_gt_path(masks_dir, stem))
        if pred.shape != gt.shape:
            # Prefer GT reference grid (feedback): nearest-neighbor resample pred.
            pred = cv2.resize(
                pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        evaluator.evaluate_multiclass_slice(
            pred,
            gt,
            slice_idx=0,
            image_id=stem,
            class_mapping=class_mapping,
        )
    return evaluator


def recompute_run_native(
    spec: RunSpec,
    *,
    corpus: str,
    repo: Path,
) -> Dict[str, Path]:
    """Write ``metrics_native_redone/`` under each site (or run root for bladder)."""
    data_root = _data_root(corpus, repo)
    masks_dir = find_combined_masks_dir(data_root)
    taxonomy = get_baseline_taxonomy_profile(data_root)
    written: Dict[str, Path] = {}

    if corpus == "prostate":
        for site in PROSTATE_TEST_SITES:
            stems = _load_test_stems(data_root, site=site)
            for stem in stems:
                if not _pred_ids_path(spec.run_dir, stem, site=site).is_file():
                    raise FileNotFoundError(
                        f"incomplete run {spec.run_name}: missing {site}/{stem}"
                    )
            evaluator = _evaluate_stems(
                stems=stems,
                run_dir=spec.run_dir,
                masks_dir=masks_dir,
                taxonomy=taxonomy,
                site=site,
            )
            out = spec.run_dir / site / "metrics_native_redone"
            out.mkdir(parents=True, exist_ok=True)
            evaluator.export_per_slice_csv(out / "per_slice_metrics.csv")
            evaluator.export_summary_csv(out / "summary_metrics.csv")
            meta = {
                "protocol": "native_gt_grid_physical_spacing",
                "eval_size": 0,
                "pixel_spacing_mm": list(taxonomy.pixel_spacing_mm),
                "method": spec.method,
                "run_name": spec.run_name,
                "site": site,
                "n_stems": len(stems),
            }
            (out / "evaluation_meta_redone.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )
            written[site] = out
    else:
        stems = _load_test_stems(data_root, site=None)
        missing = [
            s for s in stems if not _pred_ids_path(spec.run_dir, s, site=None).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{spec.method}/{spec.run_name}: missing {len(missing)}/{len(stems)} "
                f"pred_ids (e.g. {missing[:3]}); refusing partial recompute"
            )
        evaluator = _evaluate_stems(
            stems=stems,
            run_dir=spec.run_dir,
            masks_dir=masks_dir,
            taxonomy=taxonomy,
            site=None,
        )
        out = spec.run_dir / "metrics_native_redone"
        out.mkdir(parents=True, exist_ok=True)
        evaluator.export_per_slice_csv(out / "per_slice_metrics.csv")
        evaluator.export_summary_csv(out / "summary_metrics.csv")
        meta = {
            "protocol": "native_gt_grid_physical_spacing",
            "eval_size": 0,
            "pixel_spacing_mm": list(taxonomy.pixel_spacing_mm),
            "method": spec.method,
            "run_name": spec.run_name,
            "n_stems": len(stems),
        }
        (out / "evaluation_meta_redone.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        written["test"] = out
    return written


def _by_seed_dir(corpus: str, method: str, *, repo: Path) -> Path:
    return (
        corpus_results_root(corpus, DEFAULT_DATASET[corpus], repo=repo)
        / "by_seed"
        / method
    )


def _ingest_redone_prostate_run(spec: RunSpec, *, repo: Path) -> Path:
    from bapmos.results.collate_seeds import normalize_summary_rows, read_summary_metrics

    site_dirs = {
        site: spec.run_dir / site / "metrics_native_redone"
        for site in PROSTATE_TEST_SITES
        if (spec.run_dir / site / "metrics_native_redone" / "summary_metrics.csv").is_file()
    }
    if len(site_dirs) < 3:
        raise FileNotFoundError(f"expected 3 site metrics_native_redone under {spec.run_dir}")

    rows: List[Dict[str, Any]] = []
    for site, metrics_dir in site_dirs.items():
        summary = metrics_dir / "summary_metrics.csv"
        rows.extend(
            normalize_summary_rows(
                read_summary_metrics(summary),
                method=spec.method,
                run_name=spec.run_name,
                seed=spec.seed,
                site=site,
                source=str(summary),
            )
        )
    rows.extend(
        pool_organs_from_per_slice(
            site_dirs, method=spec.method, run_name=spec.run_name, seed=spec.seed
        )
    )
    dest = _by_seed_dir("prostate", spec.method, repo=repo) / f"{spec.run_name}.csv"
    return write_normalized_csv(dest, rows)


def _build_combined_redone(corpus: str, *, repo: Path) -> List[Path]:
    by_seed_root = corpus_results_root(corpus, DEFAULT_DATASET[corpus], repo=repo) / "by_seed"
    if not by_seed_root.is_dir():
        return []
    seed_rows: List[Dict[str, Any]] = []
    for method_dir in sorted(p for p in by_seed_root.iterdir() if p.is_dir()):
        for csv_path in sorted(method_dir.glob("*.csv")):
            with csv_path.open(newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    # For mean±std grouping, map run_name → experiment key like collate build
                    row = dict(r)
                    row["run_name"] = experiment_key_from_run_name(str(row.get("run_name") or ""))
                    seed_rows.append(row)

    # Prostate paper tables use site=pooled only
    if corpus == "prostate":
        seed_rows = [r for r in seed_rows if str(r.get("site") or "") == POOLED_SITE_LABEL]

    mean_rows = build_mean_std_rows(seed_rows)
    combined = corpus_results_root(corpus, DEFAULT_DATASET[corpus], repo=repo) / "combined"
    combined.mkdir(parents=True, exist_ok=True)
    ds = DEFAULT_DATASET[corpus]
    paths: List[Path] = []

    all_path = combined / f"{corpus}_{ds}_mean_std.csv"
    _write_mean_std(all_path, mean_rows)
    paths.append(all_path)

    primary = "PTV" if corpus == "prostate" else "Bladder"
    primary_rows = [r for r in mean_rows if str(r.get("organ") or "") == primary]
    organ_slug = "ptv" if corpus == "prostate" else "bladder"
    organ_path = combined / f"{corpus}_{ds}_{organ_slug}_mean_std.csv"
    _write_mean_std(organ_path, primary_rows)
    paths.append(organ_path)
    return paths


def _write_mean_std(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    from bapmos.results.collate_seeds import _mean_std_fieldnames  # noqa: PLC2701

    fields = _mean_std_fieldnames()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def run_corpus(corpus: str, *, repo: Optional[Path] = None) -> Dict[str, Any]:
    repo = repo or project_root()
    audit_path = audit_corpus(corpus, repo=repo)
    audit_rows: List[Dict[str, str]] = []
    with audit_path.open(newline="", encoding="utf-8") as f:
        audit_rows = list(csv.DictReader(f))

    complete_runs = {
        (r["method"], r["run_name"])
        for r in audit_rows
        if r.get("complete_for_native_recompute") == "1"
    }
    # A prostate run is complete only if all sites are complete
    if corpus == "prostate":
        by_run: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
        for r in audit_rows:
            by_run.setdefault((r["method"], r["run_name"]), []).append(r)
        complete_runs = {
            key
            for key, group in by_run.items()
            if len(group) == 3 and all(g.get("complete_for_native_recompute") == "1" for g in group)
        }

    specs = [s for s in _discover_runs(corpus, repo) if (s.method, s.run_name) in complete_runs]
    skipped = [
        s for s in _discover_runs(corpus, repo) if (s.method, s.run_name) not in complete_runs
    ]
    logger.info(
        "Corpus %s: %d complete runs to recompute; %d incomplete (skipped)",
        corpus,
        len(specs),
        len(skipped),
    )

    done: List[str] = []
    for spec in specs:
        logger.info("Recomputing %s / %s (seed %s)", spec.method, spec.run_name, spec.seed)
        recompute_run_native(spec, corpus=corpus, repo=repo)
        ingest_spec = RunSpec(
            method=spec.method,
            run_name=_canonical_run_name(spec.method, spec.run_name),
            run_dir=spec.run_dir,
            seed=spec.seed,
        )
        if corpus == "prostate":
            _ingest_redone_prostate_run(ingest_spec, repo=repo)
        else:
            # bladder: single-site ingest into by_seed
            from bapmos.results.collate_seeds import normalize_summary_rows, read_summary_metrics

            summary = spec.run_dir / "metrics_native_redone" / "summary_metrics.csv"
            rows = normalize_summary_rows(
                read_summary_metrics(summary),
                method=ingest_spec.method,
                run_name=ingest_spec.run_name,
                seed=ingest_spec.seed,
                site="pfus1",
                source=str(summary),
            )
            dest = (
                _by_seed_dir(corpus, ingest_spec.method, repo=repo)
                / f"{ingest_spec.run_name}.csv"
            )
            write_normalized_csv(dest, rows)
        done.append(f"{spec.method}/{spec.run_name}")

    combined_paths = _build_combined_redone(corpus, repo=repo) if done else []
    return {
        "audit": str(audit_path),
        "recomputed": done,
        "skipped": [f"{s.method}/{s.run_name}" for s in skipped],
        "combined": [str(p) for p in combined_paths],
    }


def _ingest_bladder_metrics_native_redone(spec: RunSpec, *, repo: Path) -> Path:
    from bapmos.results.collate_seeds import normalize_summary_rows, read_summary_metrics

    summary = spec.run_dir / "metrics_native_redone" / "summary_metrics.csv"
    if not summary.is_file():
        summary = spec.run_dir / "metrics" / "summary_metrics.csv"
    if not summary.is_file():
        raise FileNotFoundError(f"no native summary under {spec.run_dir}")
    ingest_name = _canonical_run_name(spec.method, spec.run_name)
    rows = normalize_summary_rows(
        read_summary_metrics(summary),
        method=spec.method,
        run_name=ingest_name,
        seed=spec.seed,
        site="pfus1",
        source=str(summary),
    )
    dest = _by_seed_dir("bladder", spec.method, repo=repo) / f"{ingest_name}.csv"
    return write_normalized_csv(dest, rows)


def ingest_exports_corpus(corpus: str, *, repo: Optional[Path] = None) -> Dict[str, Any]:
    """Build published ``by_seed/`` + ``combined/*_mean_std.csv`` from ``metrics_native_redone/``.

    Used after a native re-export with ``--eval-size 0`` (no full pred_ids required).
    """
    repo = repo or project_root()
    root = _inference_root(corpus, repo)
    done: List[str] = []
    skipped: List[str] = []
    for spec in _discover_runs(corpus, repo):
        has_native = (spec.run_dir / "metrics_native_redone" / "summary_metrics.csv").is_file()
        if not has_native:
            skipped.append(f"{spec.method}/{spec.run_name}")
            continue
        if corpus == "prostate":
            _ingest_redone_prostate_run(
                RunSpec(
                    method=spec.method,
                    run_name=_canonical_run_name(spec.method, spec.run_name),
                    run_dir=spec.run_dir,
                    seed=spec.seed,
                ),
                repo=repo,
            )
        else:
            _ingest_bladder_metrics_native_redone(spec, repo=repo)
        done.append(f"{spec.method}/{spec.run_name}")

    # Also walk *_native_redone folders that _discover_runs may already include.
    combined = _build_combined_redone(corpus, repo=repo) if done else []

    return {
        "ingested": done,
        "skipped": skipped,
        "combined": [str(p) for p in combined],
        "inference_root": str(root),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="Write pred_ids completeness audit (*_redone.csv)")
    a.add_argument("--corpus", choices=("prostate", "bladder"), required=True)

    r = sub.add_parser("run", help="Audit + native recompute for complete dumps only")
    r.add_argument("--corpus", choices=("prostate", "bladder"), required=True)

    i = sub.add_parser(
        "ingest-exports",
        help="Build by_seed/ + combined mean_std from metrics_native_redone/ (after --eval-size 0 export)",
    )
    i.add_argument("--corpus", choices=("prostate", "bladder"), required=True)

    args = p.parse_args()
    repo = project_root()
    if args.cmd == "audit":
        path = audit_corpus(args.corpus, repo=repo)
        print(path)
        return
    if args.cmd == "ingest-exports":
        result = ingest_exports_corpus(args.corpus, repo=repo)
        print(json.dumps(result, indent=2))
        return
    result = run_corpus(args.corpus, repo=repo)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
