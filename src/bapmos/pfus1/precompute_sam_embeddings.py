"""Shim — prefer ``bapmos.preprocess.bladder.precompute_sam_embeddings``."""
from bapmos.preprocess.bladder.precompute_sam_embeddings import *  # noqa: F401,F403
from bapmos.preprocess.bladder.precompute_sam_embeddings import main

if __name__ == "__main__":
    raise SystemExit(main())
