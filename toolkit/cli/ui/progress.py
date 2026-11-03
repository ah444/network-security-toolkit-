"""The progress line of a verbose scan (S16). In a module of its own since F6a
(``toolkit.cli.ui.scan`` re-exports it)."""

from __future__ import annotations

import sys
import threading


class ConsoleProgress:
    """ProgressSink for verbose mode: prints the progress line every ``interval`` seconds
    from its own thread (a slow port can't freeze it), then the final 100 % line."""

    def __init__(self, interval: float = 2) -> None:
        self.interval = interval
        self._total = 0
        self._done = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, total: int) -> None:
        """Shows the first line and starts the thread that repeats it (nothing for 0 tasks)."""
        self._total, self._done = total, 0
        self._stop.clear()
        if total > 0:  # nothing to show (and no division by zero) for 0 tasks
            self._write()
            self._thread = threading.Thread(target=self._repeat, daemon=True)
            self._thread.start()

    def advance(self, done: int = 1) -> None:
        """Counts finished checks; the scanner's worker threads call it (locked)."""
        with self._lock:
            self._done += done

    def finish(self) -> None:
        """Stops the thread and shows the last line: 100 % only if every check finished (else the
        scan was cancelled)."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        self._write()
        print("\nProgress: 100.00% complete" if self._done >= self._total else "")  # else cancelled

    def _repeat(self) -> None:
        """The thread: the line again every ``interval`` seconds until ``finish``."""
        while not self._stop.wait(self.interval):
            self._write()

    def _write(self) -> None:
        """Writes the progress line over the previous one (a carriage return)."""
        with self._lock:
            done = self._done
        sys.stdout.write(f"\rProgress: {done / self._total * 100:.2f}% complete | "
                         f"{self._total - done} tasks remaining")
        sys.stdout.flush()
