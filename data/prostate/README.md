# Place pooled prostate data here (or symlink).
# Data are not shipped — do not commit corpora.
#
# Expected subtree:
#   images/
#   masks/
#   splits_stratified/
#   spacing_contract.json
#   site_tests/
#
# Build: python -m bapmos.preprocess.prostate
# Details: docs/PREPROCESS.md
#
# Symlink example:
#   ln -s /path/to/existing/pooled pooled
