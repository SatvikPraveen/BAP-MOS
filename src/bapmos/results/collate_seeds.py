"""Ingest per-seed summary CSVs and build mean±std paper tables.

Prostate pooled training evaluates case1 / case2 / simulation separately for
clarity, then paper numbers are **slice-weighted across those sites** (same rule
as ``prostate_investigation`` ``compile_pooled_ptv``):

    pooled = mean over concatenated per-slice rows
           = sum(site_mean × n_site) / sum(n_site)

Examples::

    # Preferred: ingest a full multi-site inference run (writes sites + pooled):
    python -m bapmos.results.collate_seeds ingest-run \\
      --corpus prostate --method box --run-name pooled_seed42 \\
      --run-dir inference_output/prostate/pooled/box/box_pooled_seed42

    # Or ingest one site summary at a time (pooled row added on build if sources
    # point at per_slice_metrics.csv siblings):
    python -m bapmos.results.collate_seeds ingest \\
      --corpus prostate --method box --run-name pooled_seed42 \\
      --site simulation --summary path/to/simulation/metrics/summary_metrics.csv

    python -m bapmos.results.collate_seeds build --corpus prostate
    # → combined/prostate_pooled_mean_std.csv
    # → combined/prostate_pooled_ptv_mean_std.csv
    # bladder: combined/bladder_pfus1_mean_std.csv + bladder_pfus1_bladder_mean_std.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from bapmos.paths import project_root, resolve_under_project
from bapmos.results.layout import (
    POOLED_SITE_LABEL,
    PROSTATE_TEST_SITES,
    SEED_METRIC_COLUMNS,
    SUMMARY_COLUMN_ALIASES,
    by_seed_csv_path,
    corpus_results_root,
    experiment_key_from_run_name,
    infer_seed_from_run_name,
    list_method_dirs,
    method_by_seed_dir,
    method_combined_dir,
)

NORMALIZED_FIELDS = (
    "method",
    "run_name",
    "seed",
    "site",
    "organ",
    "dice",
    "msd_mm",
    "hd95_mm",
    "n_slices",
    "source",
)


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _pick(row: Dict[str, str], aliases: Sequence[str]) -> Optional[float]:
    for key in aliases:
        if key in row and str(row[key]).strip() != "":
            return _f(row[key])
    # case-insensitive
    lower = {k.lower(): v for k, v in row.items()}
    for key in aliases:
        if key.lower() in lower and str(lower[key.lower()]).strip() != "":
            return _f(lower[key.lower()])
    return None


def read_summary_metrics(path: Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_summary_rows(
    rows: Iterable[Dict[str, str]],
    *,
    method: str,
    run_name: str,
    seed: Optional[int] = None,
    site: str = "",
    source: str = "",
) -> List[Dict[str, Any]]:
    """Convert evaluator ``summary_metrics.csv`` rows into normalized seed rows."""
    seed_i = int(seed) if seed is not None else infer_seed_from_run_name(run_name)
    if seed_i is None:
        raise ValueError(
            f"Cannot infer seed from run_name={run_name!r}; pass --seed explicitly."
        )
    out: List[Dict[str, Any]] = []
    for row in rows:
        organ = (row.get("organ") or row.get("Organ") or "").strip() or "all"
        rec: Dict[str, Any] = {
            "method": method,
            "run_name": run_name,
            "seed": seed_i,
            "site": site or "",
            "organ": organ,
            "n_slices": _pick(row, ("n_slices",)) or "",
            "source": source,
        }
        for dest, aliases in SUMMARY_COLUMN_ALIASES.items():
            rec[dest] = _pick(row, aliases)
        out.append(rec)
    return out


def write_normalized_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(NORMALIZED_FIELDS), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") if row.get(k) is not None else "" for k in NORMALIZED_FIELDS})
    return path


def ingest_summary(
    summary_csv: Path,
    *,
    corpus: str,
    method: str,
    run_name: str,
    dataset: Optional[str] = None,
    seed: Optional[int] = None,
    site: str = "",
    repo: Optional[Path] = None,
) -> Path:
    """Write ``results/<corpus>/<dataset>/by_seed/<method>/<run_name>.csv``."""
    summary_csv = resolve_under_project(summary_csv)
    rows = normalize_summary_rows(
        read_summary_metrics(summary_csv),
        method=method,
        run_name=run_name,
        seed=seed,
        site=site,
        source=str(summary_csv),
    )
    dest = by_seed_csv_path(
        corpus, method, run_name, dataset=dataset, repo=repo  # type: ignore[arg-type]
    )
    if site:
        # One file per run still; site is a column (pooled multi-site).
        existing: List[Dict[str, Any]] = []
        if dest.is_file():
            with dest.open(newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    # Replace same site; also drop stale pooled (rebuild will refresh).
                    if (r.get("site") or "") in (site, POOLED_SITE_LABEL):
                        continue
                    existing.append(r)
        rows = existing + rows
    return write_normalized_csv(dest, rows)


def _truthy_boundary(val: Any) -> bool:
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def _mean_numeric(values: Sequence[Any]) -> Optional[float]:
    nums = [_f(v) for v in values]
    nums = [v for v in nums if v is not None]
    if not nums:
        return None
    return float(statistics.fmean(nums))


def read_per_slice_metrics(path: Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metrics_dir_from_summary_source(source: str) -> Optional[Path]:
    """``.../metrics/summary_metrics.csv`` → ``.../metrics``."""
    if not source:
        return None
    p = Path(str(source))
    if p.name == "summary_metrics.csv" and p.parent.name == "metrics":
        return p.parent
    if p.is_dir() and (p / "per_slice_metrics.csv").is_file():
        return p
    return None


def discover_site_metrics_dirs(run_dir: Path) -> Dict[str, Path]:
    """Map site → ``.../<site>/metrics`` under an inference run root."""
    run_dir = Path(run_dir)
    found: Dict[str, Path] = {}
    for site in PROSTATE_TEST_SITES:
        metrics = run_dir / site / "metrics"
        if (metrics / "per_slice_metrics.csv").is_file() or (
            metrics / "summary_metrics.csv"
        ).is_file():
            found[site] = metrics
    return found


def pool_organs_from_per_slice(
    site_metrics_dirs: Dict[str, Path],
    *,
    method: str,
    run_name: str,
    seed: int,
) -> List[Dict[str, Any]]:
    """Slice-weighted pool across sites (prostate_investigation ``compile_pooled_ptv``).

    Loads ``per_slice_metrics.csv`` per site, keeps rows with ``valid_boundary``,
    concatenates, then takes the arithmetic mean per organ. Equivalent to
    ``sum(site_mean × n_site) / total_n`` when each site mean is over its
    valid-boundary slices.
    """
    by_organ: Dict[str, List[Dict[str, str]]] = {}
    site_counts: Dict[str, int] = {}
    for site, metrics_dir in site_metrics_dirs.items():
        per_slice = Path(metrics_dir) / "per_slice_metrics.csv"
        if not per_slice.is_file():
            continue
        rows = read_per_slice_metrics(per_slice)
        kept = 0
        for row in rows:
            if "valid_boundary" in row and not _truthy_boundary(row.get("valid_boundary")):
                continue
            organ = (row.get("organ") or row.get("Organ") or "").strip()
            if not organ or organ.lower() == "all":
                continue
            by_organ.setdefault(organ, []).append(row)
            kept += 1
        site_counts[site] = kept

    if not by_organ:
        return []

    remarks = ", ".join(
        f"{s}={site_counts[s]}" for s in PROSTATE_TEST_SITES if s in site_counts
    )
    source_note = (
        f"slice_weighted_pooled valid_boundary n={sum(site_counts.values())}"
        f" ({remarks})"
    )
    out: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, str]] = []
    for organ in sorted(by_organ.keys()):
        frames = by_organ[organ]
        all_rows.extend(frames)
        out.append(
            {
                "method": method,
                "run_name": run_name,
                "seed": seed,
                "site": POOLED_SITE_LABEL,
                "organ": organ,
                "dice": _mean_numeric([r.get("dice") for r in frames]),
                "msd_mm": _mean_numeric([r.get("msd_mm") for r in frames]),
                "hd95_mm": _mean_numeric([r.get("hd95_mm") for r in frames]),
                "n_slices": float(len(frames)),
                "source": source_note,
            }
        )
    if all_rows:
        out.append(
            {
                "method": method,
                "run_name": run_name,
                "seed": seed,
                "site": POOLED_SITE_LABEL,
                "organ": "all",
                "dice": _mean_numeric([r.get("dice") for r in all_rows]),
                "msd_mm": _mean_numeric([r.get("msd_mm") for r in all_rows]),
                "hd95_mm": _mean_numeric([r.get("hd95_mm") for r in all_rows]),
                "n_slices": float(len(all_rows)),
                "source": source_note,
            }
        )
    return out


def pool_organs_from_summary_weights(
    site_rows: Sequence[Dict[str, Any]],
    *,
    method: str,
    run_name: str,
    seed: int,
) -> List[Dict[str, Any]]:
    """Fallback pool: weight each site summary mean by ``n_slices``.

    Prefer ``pool_organs_from_per_slice`` when per-slice CSVs exist; this matches
    all-slice Dice weighting but may diverge from investigation for MSD/HD95
    when ``n_slices`` ≠ ``n_valid_boundary``.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in site_rows:
        site = str(row.get("site") or "")
        if site in ("", POOLED_SITE_LABEL):
            continue
        organ = str(row.get("organ") or "")
        buckets.setdefault(organ, []).append(row)

    out: List[Dict[str, Any]] = []
    for organ, group in sorted(buckets.items()):
        weights: List[Tuple[float, Dict[str, Any]]] = []
        for r in group:
            n = _f(r.get("n_slices"))
            if n is None or n <= 0:
                continue
            weights.append((n, r))
        if not weights:
            continue
        total_n = sum(n for n, _ in weights)
        rec: Dict[str, Any] = {
            "method": method,
            "run_name": run_name,
            "seed": seed,
            "site": POOLED_SITE_LABEL,
            "organ": organ,
            "n_slices": total_n,
            "source": "slice_weighted_from_summary_n_slices",
        }
        for metric in SEED_METRIC_COLUMNS:
            num = 0.0
            den = 0.0
            for n, r in weights:
                v = _f(r.get(metric))
                if v is None:
                    continue
                num += v * n
                den += n
            rec[metric] = (num / den) if den > 0 else None
        out.append(rec)
    return out


def _site_metrics_dirs_from_seed_rows(
    seed_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for row in seed_rows:
        site = str(row.get("site") or "")
        if site not in PROSTATE_TEST_SITES:
            continue
        metrics = metrics_dir_from_summary_source(str(row.get("source") or ""))
        if metrics is not None and (metrics / "per_slice_metrics.csv").is_file():
            found[site] = metrics
    return found


def ensure_pooled_rows_for_run(
    seed_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep per-site rows; ensure one ``site=pooled`` block per (run, seed)."""
    rows = [dict(r) for r in seed_rows if str(r.get("site") or "") != POOLED_SITE_LABEL]
    sites_present = {
        str(r.get("site") or "") for r in rows if str(r.get("site") or "") in PROSTATE_TEST_SITES
    }
    if len(sites_present) < 2:
        # Single-site / bladder: nothing to pool across.
        return list(seed_rows)

    # Group by run_name + seed
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (str(r.get("run_name") or ""), str(r.get("seed") or ""))
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for (run_name, seed_s), group in sorted(groups.items()):
        out.extend(group)
        method = str(group[0].get("method") or "")
        try:
            seed_i = int(float(seed_s))
        except (TypeError, ValueError):
            seed_i = infer_seed_from_run_name(run_name) or 0
        metrics_dirs = _site_metrics_dirs_from_seed_rows(group)
        pooled: List[Dict[str, Any]] = []
        if len(metrics_dirs) >= 2:
            pooled = pool_organs_from_per_slice(
                metrics_dirs, method=method, run_name=run_name, seed=seed_i
            )
        if not pooled:
            pooled = pool_organs_from_summary_weights(
                group, method=method, run_name=run_name, seed=seed_i
            )
        out.extend(pooled)
    return out


def ingest_inference_run(
    run_dir: Path,
    *,
    corpus: str,
    method: str,
    run_name: str,
    dataset: Optional[str] = None,
    seed: Optional[int] = None,
    repo: Optional[Path] = None,
) -> Path:
    """Ingest all site summaries under a run dir and write slice-weighted pooled rows."""
    run_dir = resolve_under_project(run_dir)
    seed_i = int(seed) if seed is not None else infer_seed_from_run_name(run_name)
    if seed_i is None:
        raise ValueError(
            f"Cannot infer seed from run_name={run_name!r}; pass --seed explicitly."
        )

    site_dirs = discover_site_metrics_dirs(run_dir)
    if not site_dirs:
        raise FileNotFoundError(
            f"No site metrics under {run_dir} "
            f"(expected one of {PROSTATE_TEST_SITES} with metrics/)"
        )

    rows: List[Dict[str, Any]] = []
    for site, metrics_dir in site_dirs.items():
        summary = Path(metrics_dir) / "summary_metrics.csv"
        if not summary.is_file():
            continue
        rows.extend(
            normalize_summary_rows(
                read_summary_metrics(summary),
                method=method,
                run_name=run_name,
                seed=seed_i,
                site=site,
                source=str(summary),
            )
        )

    pooled = pool_organs_from_per_slice(
        site_dirs, method=method, run_name=run_name, seed=seed_i
    )
    if not pooled and rows:
        pooled = pool_organs_from_summary_weights(
            rows, method=method, run_name=run_name, seed=seed_i
        )
    rows.extend(pooled)

    dest = by_seed_csv_path(
        corpus, method, run_name, dataset=dataset, repo=repo  # type: ignore[arg-type]
    )
    return write_normalized_csv(dest, rows)


def _load_method_seed_rows(method_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for csv_path in sorted(method_dir.glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(dict(r))
    return rows


def _group_key(row: Dict[str, Any]) -> tuple:
    run_name = str(row.get("run_name") or "")
    return (
        str(row.get("method") or ""),
        experiment_key_from_run_name(run_name),
        str(row.get("organ") or ""),
        str(row.get("site") or ""),
    )


def build_mean_std_rows(seed_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate numeric metrics across seeds (sample std, ddof=1 when n≥2)."""
    buckets: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in seed_rows:
        buckets.setdefault(_group_key(row), []).append(row)

    out: List[Dict[str, Any]] = []
    for (method, run_name, organ, site), group in sorted(buckets.items()):
        seeds = sorted({str(r.get("seed") or "") for r in group if str(r.get("seed") or "")})
        rec: Dict[str, Any] = {
            "method": method,
            "run_name": run_name,
            "organ": organ,
            "site": site,
            "n_seeds": len(seeds),
            "seeds": ",".join(seeds),
        }
        for metric in SEED_METRIC_COLUMNS:
            vals = [_f(r.get(metric)) for r in group]
            vals = [v for v in vals if v is not None]
            if not vals:
                rec[f"{metric}_mean"] = ""
                rec[f"{metric}_std"] = ""
                rec[f"{metric}_mean_pm_std"] = ""
                continue
            mean = float(statistics.fmean(vals))
            std = float(statistics.stdev(vals)) if len(vals) >= 2 else 0.0
            rec[f"{metric}_mean"] = round(mean, 6)
            rec[f"{metric}_std"] = round(std, 6)
            rec[f"{metric}_mean_pm_std"] = f"{mean:.4f}±{std:.4f}"
        out.append(rec)
    return out


def _mean_std_fieldnames() -> List[str]:
    fields = ["method", "run_name", "organ", "site", "n_seeds", "seeds"]
    for metric in SEED_METRIC_COLUMNS:
        fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_mean_pm_std"])
    return fields


def _write_mean_std_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _mean_std_fieldnames()
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})
    return path


def _corpus_stacked_mean_std_name(corpus: str, dataset: Optional[str] = None) -> str:
    """Dedicated corpus table name, e.g. ``prostate_pooled_mean_std.csv``."""
    from bapmos.results.layout import DEFAULT_DATASET

    ds = dataset or DEFAULT_DATASET[corpus]  # type: ignore[index]
    return f"{corpus}_{ds}_mean_std.csv"


def _corpus_primary_organ_mean_std_name(
    corpus: str,
    *,
    organ_slug: str,
    dataset: Optional[str] = None,
) -> str:
    """Primary-organ paper table, e.g. ``prostate_pooled_ptv_mean_std.csv``."""
    from bapmos.results.layout import DEFAULT_DATASET

    ds = dataset or DEFAULT_DATASET[corpus]  # type: ignore[index]
    return f"{corpus}_{ds}_{organ_slug}_mean_std.csv"


def _corpus_ptv_mean_std_name(corpus: str, dataset: Optional[str] = None) -> str:
    """Backward-compatible alias for the prostate PTV primary-organ table."""
    return _corpus_primary_organ_mean_std_name(
        corpus, organ_slug="ptv", dataset=dataset
    )


# Corpus → display organ name used in by_seed rows (case-insensitive match).
_PRIMARY_ORGAN: dict[str, str] = {
    "prostate": "PTV",
    "bladder": "Bladder",
}


def collect_method_mean_std_rows(
    corpus: str,
    method: str,
    *,
    dataset: Optional[str] = None,
    repo: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load by_seed CSVs for one method → mean±std rows (no combined/ write)."""
    method_dir = method_by_seed_dir(corpus, method, dataset=dataset, repo=repo)  # type: ignore[arg-type]
    if not method_dir.is_dir():
        raise FileNotFoundError(f"No by_seed dir for method={method!r}: {method_dir}")
    seed_rows = _load_method_seed_rows(method_dir)
    if not seed_rows:
        raise FileNotFoundError(f"No seed CSVs under {method_dir}")
    if corpus == "prostate":
        seed_rows = ensure_pooled_rows_for_run(seed_rows)
    return build_mean_std_rows(seed_rows)


def _paper_rows_for_corpus(
    corpus: str, mean_rows: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Prostate paper tables use slice-weighted ``site=pooled`` only."""
    if corpus != "prostate":
        return list(mean_rows)
    pooled = [r for r in mean_rows if str(r.get("site") or "") == POOLED_SITE_LABEL]
    return pooled if pooled else list(mean_rows)


def _cleanup_legacy_combined_csvs(combined: Path, *, keep: Sequence[str]) -> None:
    """Remove intermediate method/alias CSVs; keep only the paper tables."""
    keep_set = {Path(n).name for n in keep}
    if not combined.is_dir():
        return
    for path in combined.glob("*.csv"):
        if path.name not in keep_set:
            path.unlink()


def build_corpus(
    corpus: str,
    *,
    dataset: Optional[str] = None,
    method: Optional[str] = None,
    repo: Optional[Path] = None,
) -> List[Path]:
    """Build paper tables under ``combined/``.

    Both corpora write exactly two CSVs (same strategy):
      - ``{corpus}_{dataset}_mean_std.csv`` — all organs
      - ``{corpus}_{dataset}_{primary}_mean_std.csv`` — primary organ only
        (prostate: PTV → ``…_ptv_…``; bladder: Bladder → ``…_bladder_…``)

    Prostate paper rows keep ``site=pooled`` only; bladder keeps ``site=pfus1``.
    """
    methods: List[str]
    if method:
        methods = [method]
    else:
        methods = [p.name for p in list_method_dirs(corpus, dataset=dataset, repo=repo)]  # type: ignore[arg-type]
    if not methods:
        raise FileNotFoundError(
            f"No by_seed methods under results for corpus={corpus!r}"
        )

    all_mean_rows: List[Dict[str, Any]] = []
    for mid in methods:
        all_mean_rows.extend(
            collect_method_mean_std_rows(
                corpus, mid, dataset=dataset, repo=repo
            )
        )

    paper_rows = _paper_rows_for_corpus(corpus, all_mean_rows)
    combined = method_combined_dir(corpus, dataset=dataset, repo=repo)  # type: ignore[arg-type]
    combined.mkdir(parents=True, exist_ok=True)

    full_name = _corpus_stacked_mean_std_name(corpus, dataset)
    written: List[Path] = [_write_mean_std_csv(combined / full_name, paper_rows)]

    primary = _PRIMARY_ORGAN.get(corpus)
    if primary:
        primary_rows = [
            r
            for r in paper_rows
            if str(r.get("organ") or "").strip().lower() == primary.lower()
        ]
        organ_slug = primary.lower().replace(" ", "_")
        primary_name = _corpus_primary_organ_mean_std_name(
            corpus, organ_slug=organ_slug, dataset=dataset
        )
        written.append(_write_mean_std_csv(combined / primary_name, primary_rows))
        _cleanup_legacy_combined_csvs(
            combined, keep=[full_name, primary_name]
        )
    else:
        _cleanup_legacy_combined_csvs(combined, keep=[full_name])

    return written


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Collate per-seed test metrics into mean±std paper tables."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Ingest one summary_metrics.csv into by_seed/")
    ing.add_argument("--corpus", choices=("prostate", "bladder"), required=True)
    ing.add_argument("--dataset", default=None, help="Default: pooled | pfus1")
    ing.add_argument("--method", required=True, help="Stable method id (box, bapmos, …)")
    ing.add_argument("--run-name", required=True)
    ing.add_argument("--seed", type=int, default=None)
    ing.add_argument("--site", default="", help="Optional pooled site (simulation/case1/case2)")
    ing.add_argument("--summary", required=True, help="Path to summary_metrics.csv")

    run = sub.add_parser(
        "ingest-run",
        help="Ingest simulation/case1/case2 under an inference run + slice-weighted pooled",
    )
    run.add_argument("--corpus", choices=("prostate", "bladder"), required=True)
    run.add_argument("--dataset", default=None)
    run.add_argument("--method", required=True)
    run.add_argument("--run-name", required=True)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument(
        "--run-dir",
        required=True,
        help="Inference run root containing simulation/, case1/, case2/",
    )

    bld = sub.add_parser("build", help="Build combined per_seed + mean_std CSVs")
    bld.add_argument("--corpus", choices=("prostate", "bladder"), required=True)
    bld.add_argument("--dataset", default=None)
    bld.add_argument("--method", default=None, help="If omitted, build all methods under by_seed/")

    args = p.parse_args(argv)
    repo = project_root()

    if args.cmd == "ingest":
        dest = ingest_summary(
            Path(args.summary),
            corpus=args.corpus,
            method=args.method,
            run_name=args.run_name,
            dataset=args.dataset,
            seed=args.seed,
            site=args.site,
            repo=repo,
        )
        print(f"Wrote {dest}")
        print(f"Corpus root: {corpus_results_root(args.corpus, args.dataset, repo=repo)}")
        return

    if args.cmd == "ingest-run":
        dest = ingest_inference_run(
            Path(args.run_dir),
            corpus=args.corpus,
            method=args.method,
            run_name=args.run_name,
            dataset=args.dataset,
            seed=args.seed,
            repo=repo,
        )
        print(f"Wrote {dest}")
        print(f"Corpus root: {corpus_results_root(args.corpus, args.dataset, repo=repo)}")
        return

    if args.cmd == "build":
        paths = build_corpus(
            args.corpus,
            dataset=args.dataset,
            method=args.method,
            repo=repo,
        )
        for path in paths:
            print(f"Wrote {path}")
        return


if __name__ == "__main__":
    main()
