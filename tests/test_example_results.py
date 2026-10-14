"""results/example/ mirrors what the collation writes, with placeholders instead of values."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from bapmos.results.collate_seeds import NORMALIZED_FIELDS, _mean_std_fieldnames

EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "results" / "example"
EXAMPLE_CSVS = sorted(EXAMPLE_ROOT.rglob("*.csv"))
# A measured value would appear as a decimal number; seeds and n_seeds are plain integers.
DECIMAL = re.compile(r"^-?\d+\.\d+$")


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def test_examples_cover_both_corpora():
    combined = {p.name for p in EXAMPLE_CSVS if p.parent.name == "combined"}
    assert combined == {
        "prostate_pooled_mean_std.csv",
        "prostate_pooled_ptv_mean_std.csv",
        "bladder_pfus1_mean_std.csv",
        "bladder_pfus1_bladder_mean_std.csv",
    }
    assert any("by_seed" in p.parts for p in EXAMPLE_CSVS)


@pytest.mark.parametrize("path", EXAMPLE_CSVS, ids=lambda p: p.relative_to(EXAMPLE_ROOT).as_posix())
def test_example_matches_collation_columns_and_holds_no_values(path):
    fields, rows = _read(path)
    expected = _mean_std_fieldnames() if path.parent.name == "combined" else list(NORMALIZED_FIELDS)
    assert fields == expected
    assert rows, "an example table should show its rows"
    for row in rows:
        values = [v for k, v in row.items() if k != "source"]
        assert not any(DECIMAL.match(v) for v in values), f"numeric value in {path.name}: {row}"
