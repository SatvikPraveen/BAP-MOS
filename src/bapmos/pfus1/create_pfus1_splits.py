"""Shim — prefer ``bapmos.preprocess.bladder.create_splits``."""
from bapmos.preprocess.bladder.create_splits import *  # noqa: F401,F403
from bapmos.preprocess.bladder.create_splits import main

if __name__ == "__main__":
    raise SystemExit(main())
