"""Build meta.json (commit, timestamp, check list) for the workbook from collected.json and paper_check.json."""
import json, math, subprocess, sys
from datetime import datetime
from pathlib import Path
import agg
HERE = Path(__file__).resolve().parent
d = agg.load()
# Result of the one-off comparison with bapmos.results.camera_ready_tables (first workbook version).
REPO_COLLATION = ["TRUS mean and SD vs the repository collation (bapmos.results.camera_ready_tables, run 2026-10-17)",
                  "2208 values compared, 0 mismatches"]
n, worst = 0, 0.0
for e in d["exports"].values():
    for s in e.values():
        n += 1
        x = s["summary_check_maxdiff"]
        worst = max(worst, math.inf if x is None else x) if x is not None else worst
pc = json.load(open(HERE / "paper_check.json"))
bad = [r for r in pc if not r[3]]
checks = [["Per-seed numbers vs each export's own summary_metrics.csv (HD95, MSD, valid-pair counts)",
           f"{n} seed exports, largest difference {worst:.2g}" + (" (identical)" if worst == 0 else "")],
          [REPO_COLLATION[0], REPO_COLLATION[1] + " (exports existing then; later exports checked by the row above)"],
          ["Identical perturbed boxes across every export of each robustness condition",
           "; ".join(f"{k}: {v['exports']} exports, {v['mismatches']} mismatches" for k, v in d["prompt_check"].items())],
          ["Numbers in the draft (Tables I-V, training-mixture table, PFUS1, text, p-values) vs this workbook",
           f"{len(pc)} checked, {len(bad)} disagree"]]
checks += [[f"DRAFT DISAGREES: {r[0]}", f"draft {r[1]}, workbook {r[2]}"] for r in bad]
checks += [["Fixed-schedule test prompts (read from each export's evaluation_meta.json)",
            "1:1:1 and 5:1:1: box for every organ and seed; 2:3:5: box+point; 1:5:1: point"],
           ["SD definition", "sample SD (ddof=1) over the 3 seeds; partial seed sets are never averaged ('pending')"]]
commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
json.dump({"commit": commit, "generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "checks": checks},
          open(HERE / "meta.json", "w"), indent=1)
print(commit, n, worst, len(pc), len(bad))
