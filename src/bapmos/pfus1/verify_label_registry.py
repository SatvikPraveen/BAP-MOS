"""Shim — prefer ``bapmos.preprocess.bladder.verify_label_registry``."""
from bapmos.preprocess.bladder.verify_label_registry import *  # noqa: F401,F403
from bapmos.preprocess.bladder.verify_label_registry import main

if __name__ == "__main__":
    raise SystemExit(main())
