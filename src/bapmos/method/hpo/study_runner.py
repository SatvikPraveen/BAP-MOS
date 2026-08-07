"""Compatibility shim — canonical implementation: ``bapmos.hpo.study_runner``."""

from bapmos.hpo.study_runner import *  # noqa: F401,F403
from bapmos.hpo.study_runner import main

if __name__ == "__main__":
    main()
