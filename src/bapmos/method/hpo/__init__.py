"""Compatibility package — canonical HPO lives in ``bapmos.hpo``.

Import from ``bapmos.hpo`` in new code. Submodules here re-export the same
objects so older ``bapmos.method.hpo.*`` imports keep working.
"""

from bapmos.hpo.search_space import (
    searched_clip_scale_organ_baseline_overrides,
    trial_overrides,
)

__all__ = ["trial_overrides", "searched_clip_scale_organ_baseline_overrides"]
