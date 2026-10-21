"""Record the test-time boxes an export used, so paired robustness runs can be checked.

Robustness exports perturb each box with offsets that depend only on (slice, class id,
mode, n) (``prompt_geometry.perturb_box_paired``). Every model family writes the boxes it
actually used to ``<output_dir>/prompts_used.json`` as ``{slice: {class_id: [x0, y0, x1, y1]}}``;
:func:`compare_prompt_logs` confirms two exports received the same boxes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

PROMPT_LOG_NAME = "prompts_used.json"


def write_prompt_log(log: Mapping[str, Mapping[str, List[int]]], output_dir: Path, meta: Mapping) -> Path:
    """Write ``log`` with its perturbation settings next to the export."""
    path = Path(output_dir) / PROMPT_LOG_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": dict(meta), "boxes": log}, indent=0, sort_keys=True))
    return path


def compare_prompt_logs(a: Path, b: Path) -> Tuple[int, int, List[str]]:
    """Return (shared boxes, mismatched boxes, example mismatches) for two prompt logs.

    Boxes are compared on the (slice, class id) pairs both exports prompted; a pair present
    in only one export is counted as a mismatch, since paired exports prompt the same organs.
    """
    la: Dict = json.loads(Path(a).read_text())["boxes"]
    lb: Dict = json.loads(Path(b).read_text())["boxes"]
    shared = mismatched = 0
    examples: List[str] = []
    for key in sorted(set(la) | set(lb)):
        ba, bb = la.get(key, {}), lb.get(key, {})
        for cid in sorted(set(ba) | set(bb)):
            if ba.get(cid) == bb.get(cid):
                shared += 1
            else:
                mismatched += 1
                if len(examples) < 5:
                    examples.append(f"{key} class {cid}: {ba.get(cid)} vs {bb.get(cid)}")
    return shared, mismatched, examples
