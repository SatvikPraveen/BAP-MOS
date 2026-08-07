# Place PFUS1 bladder data here (or symlink).
# Data are not shipped — do not commit corpora.
#
# Expected subtree under data/bladder/pfus1/:
#   masks/
#   splits_*   (e.g. splits_patient_70_15_15_seed42/)
#   label registry / report metadata
#
# Raw PNG+JSON (preferred):
#   data/bladder/pfus1_raw/Pxxx/frame_*.png +.json
#
# Build: python -m bapmos.preprocess.bladder
# Details: docs/PREPROCESS.md
#
# Symlink examples (from data/bladder/):
#   ln -s /path/to/existing/pfus1_bundle pfus1
#   ln -s /path/to/existing/pfus1_raw pfus1_raw
