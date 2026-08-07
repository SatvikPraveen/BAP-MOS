"""Shim — prefer ``bapmos.preprocess.bladder.analyze_dataset``."""
from bapmos.preprocess.bladder.analyze_dataset import *  # noqa: F401,F403
from bapmos.preprocess.bladder.analyze_dataset import main

if __name__ == "__main__":
    raise SystemExit(main())
