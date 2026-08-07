"""Map deprecated ``bapmos.<pkg>`` imports onto ``bapmos.legacy.<pkg>`` (same module object)."""

from __future__ import annotations

import importlib
import sys
from typing import Dict, Optional

# Root package name → canonical legacy package.
ALIASES: Dict[str, str] = {
    "bapmos.optimization": "bapmos.legacy.optimization",
    "bapmos.singleorgan": "bapmos.legacy.singleorgan",
    "bapmos.ablations": "bapmos.legacy.ablations",
    "bapmos.pfus1_advanced": "bapmos.legacy.pfus1_advanced",
}


def _canonical_name(fullname: str) -> Optional[str]:
    for alias, canonical in ALIASES.items():
        if fullname == alias or fullname.startswith(alias + "."):
            return canonical + fullname[len(alias) :]
    return None


class _LegacyAliasFinder:
    """
    Alias finder using the legacy ``load_module`` hook.

    ``find_spec`` + ``create_module`` still dual-loads under some PathFinder
    interactions when on-disk shim packages exist; ``load_module`` returns the
    canonical module object and keeps ``sys.modules`` identity.
    """

    def find_module(self, fullname, path=None):  # noqa: ARG002
        if _canonical_name(fullname) is not None:
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        real = _canonical_name(fullname)
        if real is None:
            raise ImportError(fullname)
        mod = importlib.import_module(real)
        sys.modules[fullname] = mod
        return mod


def install() -> None:
    """Idempotent: put the alias finder first on ``sys.meta_path``."""
    if any(isinstance(f, _LegacyAliasFinder) for f in sys.meta_path):
        return
    sys.meta_path.insert(0, _LegacyAliasFinder())


def alias_module(alias_name: str) -> None:
    """Bind ``sys.modules[alias_name]`` to its canonical legacy module."""
    canonical = ALIASES.get(alias_name)
    if canonical is None:
        raise KeyError(f"No legacy alias for {alias_name!r}")
    install()
    mod = importlib.import_module(canonical)
    sys.modules[alias_name] = mod
