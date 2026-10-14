"""Map deprecated ``bapmos.<pkg>`` imports onto ``bapmos.legacy.<pkg>`` (same module object)."""

from __future__ import annotations

import importlib
import importlib.util
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
    Meta-path finder that resolves a deprecated name to the canonical module object.

    ``create_module`` returns the already-imported canonical module and ``exec_module``
    does nothing, so ``bapmos.optimization.metrics is bapmos.legacy.optimization.metrics``
    and classes stay identical (``isinstance`` keeps working across both names).
    """

    def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
        """Return a spec served by this finder for aliased names; None for everything else."""
        if _canonical_name(fullname) is None:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        """Import and return the canonical module in place of a new module object."""
        return importlib.import_module(_canonical_name(spec.name))

    def exec_module(self, module):  # noqa: ARG002
        """The canonical module is already executed."""


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
