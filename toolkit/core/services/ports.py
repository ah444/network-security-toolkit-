"""The toolkit's ports (plan §2): interfaces only where an implementation really is swapped
or faked in tests. There are exactly four; everything else is plain functions and dict
registries.

- ``ModelStore``: where trained models are kept (signed bundles on disk:
  ``toolkit.core.services.ids.exports.BundleStore``).
- ``IntelProvider``: an online (or offline, or fake) source of facts about a detected
  product or an IP address (NVD, OpenRouter, AbuseIPDB).
- ``ProgressSink``: where a long task reports progress (the console, a web job, a test).
- ``ScanEngine``: what checks a host's ports (the toolkit's own scanner,
  ``toolkit.core.services.recon.engines.BuiltinEngine``; nmap, when installed, steps N2-N4).

Ports are structural (``typing.Protocol``): an adapter only needs the right methods.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # types only: the ports themselves import no layer
    from .ids.exports import BundleInfo, SavedBundle
    import threading

    from ..data.recon.findings import IntelFinding
    from ..data.recon.scanner import PortResult
    from .ids.bundle import Bundle
    from .ids.runs import TrainingRun


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


@runtime_checkable
class IntelProvider(Protocol):
    """An online, offline or fake source of facts.

    ``kind`` says what it answers: ``"cves"`` and ``"explanation"`` are asked about a
    detected product and version (e.g. ``OpenSSH 8.9``, never a raw banner),
    ``"ip_reputation"`` about an IP address. ``name`` labels its failures in reports.
    """

    name: str
    kind: str

    def lookup(self, subject: str) -> IntelFinding:
        """What the provider found about ``subject``; raises on any failure (network,
        timeout, bad reply), which the caller records."""


@runtime_checkable
class ProgressSink(Protocol):
    """Where a long task reports how far it is: ``start(total)``, then ``advance()`` per
    finished unit (also from worker threads, so implementations must be thread-safe),
    then ``finish()`` once, also after an error."""

    def start(self, total: int) -> None:
        """``total`` units are about to be done (0: there is nothing to show)."""

    def advance(self, done: int = 1) -> None:
        """``done`` more units finished; may be called from several threads at once."""

    def finish(self) -> None:
        """The task ended (also after an error or a cancel); called once."""


@runtime_checkable
class ScanEngine(Protocol):
    """Checks a host's ports (step N1). ``name`` picks it; ``available`` says why it can't run on
    this computer (None: it can). ``scan`` gives every port of ``host`` for each protocol,
    sorted by port, reports each finished check to ``progress`` and stops once ``stop`` is set
    (``toolkit.core.data.recon.scanner.ScanStopped``). ``os_matches``: what it found of the
    host's OS in its last scan, (name, accuracy %) best first (step N4; () if it doesn't look)."""

    name: str
    os_matches: tuple[tuple[str, int], ...]

    def available(self) -> str | None:
        """None when this engine can run here, else why not (to show the user)."""

    def scan(self, host: str, ports: Iterable[int], protocols: Sequence[str],
             progress: ProgressSink | None = None,
             stop: threading.Event | None = None) -> list[PortResult]:
        """The port checks of ``host`` (an address already checked and authorised)."""


class NullProgress:
    """A ProgressSink that reports nothing (the default)."""

    def start(self, total: int) -> None:
        """Does nothing."""
        pass

    def advance(self, done: int = 1) -> None:
        """Does nothing."""
        pass

    def finish(self) -> None:
        """Does nothing."""
        pass
