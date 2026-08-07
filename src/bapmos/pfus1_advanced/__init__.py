"""Deprecated shim — prefer ``bapmos.legacy.pfus1_advanced``.

Imports are aliased to the legacy package (same module objects) via
``bapmos._legacy_aliases``. This file is a fallback if the finder is not yet
installed.
"""
from bapmos._legacy_aliases import alias_module

alias_module(__name__)
