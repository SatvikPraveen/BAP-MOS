"""
Verify PFUS1 JSON label → class ID order matches ``bapmos.data.organ_registry`` and ``constants``.

Run from repo root::

    python -m bapmos.preprocess.bladder.verify_label_registry
"""

from __future__ import annotations

import sys

from bapmos.data.organ_registry import PFUS1_EIGHT_ORGANS, PFUS1_ORGAN_TO_CLASS
from bapmos.preprocess.bladder.constants import JSON_LABEL_TO_CLASS_ID, PFUS1_ALL_LABELS


def main(argv: list[str] | None = None) -> int:
    del argv  # CLI takes no flags; signature matches other preprocess mains.
    errors: list[str] = []

    for json_label, cid in PFUS1_ALL_LABELS:
        reg_cid = JSON_LABEL_TO_CLASS_ID.get(json_label)
        if reg_cid != cid:
            errors.append(f"constants mismatch for {json_label!r}: {cid} vs {reg_cid}")

    for organ in PFUS1_EIGHT_ORGANS:
        key_cid = PFUS1_ORGAN_TO_CLASS.get(organ.key)
        if key_cid != organ.class_id:
            errors.append(
                f"organ_registry key {organ.key!r}: class_id {organ.class_id} vs map {key_cid}"
            )

    # JSON display names may differ from snake_case organ keys / evaluator labels;
    # we only require class_id alignment between constants and the registry.
    expected_order = [cid for _, cid in PFUS1_ALL_LABELS]
    registry_order = [o.class_id for o in PFUS1_EIGHT_ORGANS]
    if expected_order != registry_order:
        errors.append(
            f"class ID order mismatch:\n  constants: {expected_order}\n  registry:  {registry_order}"
        )

    if errors:
        print("PFUS1 label registry check FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("PFUS1 label registry check PASSED (constants ↔ organ_registry, IDs 1–8).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
