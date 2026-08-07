"""Shim — prefer ``bapmos.preprocess.bladder.visualize_samples``."""
from bapmos.preprocess.bladder.visualize_samples import *  # noqa: F401,F403
from bapmos.preprocess.bladder.visualize_samples import main

if __name__ == "__main__":
    raise SystemExit(main())
