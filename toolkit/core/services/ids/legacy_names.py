"""The module names that older bundles refer to (plan D11).

A bundle's pickle names every class and function by its module: ``FittedModel`` and the
pipeline's functions, the feature selector, ... Two older generations of names exist:
``toolkit.ids_core.<module>`` (bundles saved before v5.1 step 1.5d) and
``toolkit.services.ids.<module>`` (v5.1 step 1.5d to v5.2 step 2.3a). While such a bundle is
unpickled, those names point to the modules they are now (``toolkit.core.services.ids``);
before and after, the old names don't exist, so no code can keep importing them
(``tests/test_architecture.py`` also refuses them).
"""

from __future__ import annotations

import importlib
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager

NEW = "toolkit.core.services.ids"
# toolkit/ids_core (v5.1 step 1.4); toolkit/services/ids added prediction (step 1.5d)
MODULES = ("balancing", "batch", "bundle", "datasets", "evaluation", "exports", "leakage",
           "pipeline", "preview", "profiles", "registry", "runs", "sampling", "split")
V51_MODULES = (*MODULES, "prediction")
OLD_TO_NEW = {"toolkit.ids_core": NEW,
              **{f"toolkit.ids_core.{name}": f"{NEW}.{name}" for name in MODULES},
              "toolkit.services.ids": NEW,
              **{f"toolkit.services.ids.{name}": f"{NEW}.{name}" for name in V51_MODULES}}

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
