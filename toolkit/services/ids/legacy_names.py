"""The module names that bundles saved before v5.1 step 1.5d refer to (plan D11).

A bundle's pickle names every class and function by its module: ``FittedModel`` and the
pipeline's functions, the feature selector, ... Bundles saved before the layered layout
name ``toolkit.ids_core.<module>``. While such a bundle is unpickled, those names point to
the modules they are now; before and after, ``toolkit.ids_core`` doesn't exist, so no code
can keep importing it (``tests/test_architecture.py`` also refuses it).
"""

from __future__ import annotations

import importlib
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager

MODULES = ("balancing", "batch", "bundle", "datasets", "evaluation", "exports", "leakage",
           "pipeline", "preview", "profiles", "registry", "runs", "sampling", "split")
OLD_TO_NEW = {"toolkit.ids_core": "toolkit.services.ids",
              **{f"toolkit.ids_core.{name}": f"toolkit.services.ids.{name}" for name in MODULES}}

_lock = threading.Lock()
_users = 0  # bundles being unpickled right now (web requests run in threads)


@contextmanager
def old_module_names() -> Iterator[None]:
    """Make the old module names importable for the duration of the ``with`` block."""
    global _users
    with _lock:
        if _users == 0:
            for old, new in OLD_TO_NEW.items():
                sys.modules[old] = importlib.import_module(new)
        _users += 1
    try:
        yield
    finally:
        with _lock:
            _users -= 1
            if _users == 0:
                for old in OLD_TO_NEW:
                    sys.modules.pop(old, None)
