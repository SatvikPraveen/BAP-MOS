"""Read the saved workbook back and recompute every number from collected.json."""
import math
import re
import statistics as st
import sys
from pathlib import Path

from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agg  # noqa: E402

d = agg.load()
wb = load_workbook(sys.argv[1])
problems, n_num, n_fmt, n_pend = [], 0, 0, 0
REV = {"Dice": "dice", "HD95": "hd95_mm", "MSD": "msd_mm"}

# 1. No broken values anywhere.
for ws in wb.worksheets:
    for row in ws.iter_rows():
        for c in row:
            v = c.value
            if isinstance(v, float) and not math.isfinite(v):
                problems.append(f"{ws.title}!{c.coordinate} non-finite {v}")
            if isinstance(v, str) and re.search(r"\b(nan|None|inf|NaN)\b", v):
                problems.append(f"{ws.title}!{c.coordinate} text '{v}'")
            if isinstance(v, str) and v.startswith("="):
                problems.append(f"{ws.title}!{c.coordinate} formula '{v}'")

# 2. Long sheet: recompute mean, SD and per-seed values from the raw per-seed numbers.
ws = wb["All results (long)"]
HR = next(c.row for c in ws["A"] if c.value == "Dataset")
hdr = [c.value for c in ws[HR]]
ix = {h: i for i, h in enumerate(hdr)}
keymap = {}
for key in d["exports"]:
    keymap.setdefault(key, key)
rows = list(ws.iter_rows(min_row=HR + 1, values_only=True))
expected_rows = sum(len(agg.METRICS) * (5 if k.startswith("TRUS") else 9) for k, e in d["exports"].items()
                    if e or "|superseded|" not in k)
if len(rows) != expected_rows:
    problems.append(f"long sheet has {len(rows)} rows, expected {expected_rows}")
for r in rows:
    vals = [x for x in (r[ix["Seed 42"]], r[ix["Seed 43"]], r[ix["Seed 44"]]) if x is not None]
    if vals and len(vals) != r[ix["Seeds finished"]]:
        problems.append(f"long: seed count mismatch {r[:6]}")
    if len(vals) == 3:
        n_num += 1
        m, s = st.fmean(vals), st.stdev(vals)
        if abs(m - r[ix["Mean"]]) > 1e-12 or abs(s - r[ix["SD"]]) > 1e-12:
            problems.append(f"long: mean/sd mismatch {r[:6]}")
        k = 3 if (r[ix["Metric"]] == "Dice" or r[ix["Dataset"]] == "TRUS") else 2
        if r[ix["Mean ± SD"]] != f"{m:.{k}f} ± {s:.{k}f}":
            problems.append(f"long: formatted mismatch {r[:6]} {r[ix['Mean ± SD']]}")
    elif r[ix["Seeds finished"]] == 3 and not vals:
        if r[ix["Mean ± SD"]] != "n/a" or "single_" not in str(r[ix["Export (seed 42)"]]):
            problems.append(f"long: unexpected n/a {r[:6]}")
    else:
        n_pend += 1
        if r[ix["Mean"]] is not None or r[ix["Mean ± SD"]] != "pending":
            problems.append(f"long: partial seeds shown as a value {r[:6]}")

# 3. Formatted table sheets: every 'm ± s' cell must equal the long sheet's value for that row/column.
lookup = {(r[ix["Dataset"]], r[ix["Boxes"]], r[ix["Model"]], r[ix["Organ"]], r[ix["Metric"]]): r[ix["Mean ± SD"]]
          for r in rows}
TABLES = {"TRUS exact boxes": ("TRUS", None), "TRUS ablations": ("TRUS", None), "TRUS robustness": ("TRUS", "col"),
          "TRUS LOCO + U-Net boxes": ("TRUS", "col"), "TRUS Table I rerun": ("TRUS", None),
          "PFUS1 exact boxes": ("PFUS1", None), "PFUS1 robustness": ("PFUS1", "col"), "TRUS superseded": ("TRUS", "col")}
for title, (ds, boxes_col) in TABLES.items():
    ws = wb[title]
    hrow = next(i for i, row in enumerate(ws.iter_rows(values_only=True), 1) if row[0] in ("Model", "Boxes", "Setting"))
    hdr = [c.value for c in ws[hrow]]
    first = 2 if boxes_col else 1
    for row in ws.iter_rows(min_row=hrow + 1, values_only=True):
        boxes = row[0] if boxes_col else "Exact boxes"
        model = row[first - 1]
        for j in range(first, len(hdr)):
            h = hdr[j]
            mt = re.match(r"^(.*) (Dice|HD95|MSD)(?: \((?:mm|px)\))?$", h)
            organ, metric = mt.group(1), mt.group(2)
            # ablations/table1 sheets come from the same 'exact' block; Table I rerun is its own block
            want = None
            for (dds, b, mm, o, me), v in lookup.items():
                if dds == ds and b == boxes and mm == model and o == organ and me == metric:
                    if title == "TRUS Table I rerun" or want is None:
                        want = v
            if want is None:
                problems.append(f"{title}: no long-sheet row for {boxes}/{model}/{organ}/{metric}")
            elif row[j] != want:
                problems.append(f"{title}: {boxes}/{model}/{organ}/{metric} shows {row[j]}, long sheet {want}")
            n_fmt += 1

# 4. Paired-test sheet: values equal the collected paired rows.
ws = wb["Paired tests"]
PR = next(c.row for c in ws["A"] if c.value == "Dataset")
prow = [r for r in ws.iter_rows(min_row=PR + 1, values_only=True)]
exp = sum(len(p["rows"]) if p["status"] == "ok" else 1 for p in d["paired"].values())
if len(prow) != exp:
    problems.append(f"paired sheet {len(prow)} rows, expected {exp}")
flat = []
for p in d["paired"].values():
    if p["status"] == "ok":
        flat += p["rows"]
        continue
    flat.append(None)
for r, src in zip(prow, flat):
    if src is None:
        if r[-1] != "pending":
            problems.append(f"paired: pending row not marked {r[:4]}")
        continue
    for col, f in ((7, "n_pairs"), (12, "p_wilcoxon"), (13, "p_holm"), (11, "a_better_frac")):
        a, b = r[col], src[f]
        if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-15):
            problems.append(f"paired: {r[:7]} {f} {a} vs {b}")

print(f"numeric cells recomputed: {n_num}; formatted table cells checked: {n_fmt}; "
      f"pending rows: {n_pend}; paired rows: {len(prow)}")
print("PROBLEMS:", len(problems))
for p in problems[:30]:
    print(" ", p)
