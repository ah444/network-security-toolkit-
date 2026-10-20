"""The toolkit's ports (plan §2): interfaces only where an implementation really is swapped
or faked in tests. There are exactly three; everything else is plain functions and dict
registries.

- ``ModelStore``: where trained models are kept (signed bundles on disk:
  ``toolkit.bundle_files.BundleStore``).
- ``IntelProvider``: an online (or offline, or fake) source of facts about a detected
  product or an IP address (NVD, OpenRouter, AbuseIPDB).
- ``ProgressSink``: where a long task reports progress (the console, a web job, a test).

Ports are structural (``typing.Protocol``): an adapter only needs the right methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # types only: the ports themselves import no layer
    from .bundle_files import BundleInfo, SavedBundle
    from .ids_core.bundle import Bundle
    from .ids_core.runs import TrainingRun


@runtime_checkable
class ModelStore(Protocol):
    """Saves trained models and loads them back; every load checks the file first."""

    def save(self, run: TrainingRun) -> SavedBundle:
        """Store ``run``'s models; never replaces an existing one (FileExistsError)."""

    def list_bundles(self) -> list[BundleInfo]:
        """What the store holds, newest first (read without loading any model)."""

    def load(self, path: str | Path) -> Bundle:
        """A stored model, refused (BundleError) before deserializing if it can't be trusted."""

    def open(self, data: bytes) -> Bundle:
        """``load`` for bytes already in memory (e.g. an upload)."""
