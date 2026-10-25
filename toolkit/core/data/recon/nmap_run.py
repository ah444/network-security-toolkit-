"""Running nmap (step N3): its XML read as nmap writes it, the progress from its
``<taskprogress>`` elements, a stop or Ctrl+C that ends it (asked to end, killed after
``KILL_AFTER`` seconds), and its failures in plain words.

nmap runs by its full path from its own folder, with no shell and standard input closed. Its
XML is kept up to ``MAX_XML_BYTES`` (more ends the run), what it writes to standard error up
to ``MAX_ERROR_BYTES``, for the message if it fails. The command is ``nmap.nmap_command``'s;
the XML is read by ``nmap_xml``.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO

from .nmap import NEEDS_PRIVILEGES
from .scanner import ScanStopped
from .text import one_line

MAX_XML_BYTES = 50 * 1024 * 1024
MAX_ERROR_BYTES = 64 * 1024
KILL_AFTER = 5  # seconds between asking nmap to end and killing it
POLL = 0.2  # seconds between looks at the stop event
TASK_PROGRESS = re.compile(
    rb'<taskprogress task="([^"]{1,60})"[^>]*? percent="(\d{1,3}(?:\.\d{1,2})?)"')
KNOWN_ERRORS = (  # what nmap writes to standard error -> what the user is told
    (re.compile(r"requires root privileges|requires raw socket", re.IGNORECASE),
     NEEDS_PRIVILEGES),
    (re.compile(r"npcap|winpcap|pcap_open|dnet: failed to open", re.IGNORECASE),
     "nmap could not use Npcap, which it needs for raw packets: reinstall nmap with Npcap "
     "(https://nmap.org/download), or scan TCP only without OS detection."),
)

Progress = Callable[[str, float], None]  # (nmap's task, its percent), from a reader thread


class NmapFailed(Exception):
    """nmap ran but gave no result; the message says why in plain words."""


def explain(returncode: int, stderr: bytes) -> str:
    """Why nmap failed: a known problem in plain words (``KNOWN_ERRORS``), else its exit code
    and its first three lines of error, cleaned (they may quote text from the network)."""
    text = stderr.decode("utf-8", "replace")
    for pattern, message in KNOWN_ERRORS:
        if pattern.search(text):
            return message
    lines = [one_line(line, 200) for line in text.splitlines()
             if line.strip() and not line.startswith("QUITTING")]
    return f"nmap stopped with exit code {returncode}: {'; '.join(lines[:3]) or 'no message'}"


class _Reader(threading.Thread):
    """Reads one of nmap's outputs to its end, keeping at most ``limit`` bytes (``overflow``
    once more came); each whole line goes to ``on_line``."""

    def __init__(self, stream: IO[bytes], limit: int,
                 on_line: Callable[[bytes], None] | None = None) -> None:
        super().__init__(daemon=True)
        self.stream, self.limit, self.on_line = stream, limit, on_line
        self.data = bytearray()
        self.overflow = False

    def run(self) -> None:
        """Reads until nmap closes the output; a line longer than 4 KB is cut for ``on_line``."""
        pending = b""
        read = getattr(self.stream, "read1", self.stream.read)  # what is there, not 64 KB
        for chunk in iter(lambda: read(65536), b""):
            if self.overflow or len(self.data) + len(chunk) > self.limit:
                self.overflow = True
            else:
                self.data += chunk
            if self.on_line:
                *lines, pending = (pending + chunk).split(b"\n")
                pending = pending[-4096:]
                for line in lines:
                    self.on_line(line)


def _end(process: subprocess.Popen[bytes]) -> None:
    """Asks nmap to end, kills it ``KILL_AFTER`` seconds later; nothing if it has ended."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(KILL_AFTER)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _progress_lines(progress: Progress | None) -> Callable[[bytes], None] | None:
    """A line reader that gives ``progress`` each ``<taskprogress>`` element's task and percent."""
    if progress is None:
        return None

    def on_line(line: bytes) -> None:
        match = TASK_PROGRESS.search(line)
        if match:
            progress(match.group(1).decode("ascii", "replace"), float(match.group(2)))
    return on_line


def run_nmap(command: Sequence[str], progress: Progress | None = None,
             stop: threading.Event | None = None, time_limit: float | None = None) -> bytes:
    """Runs ``command`` (nmap's full path first) and gives its XML.

    A set ``stop``, Ctrl+C, ``time_limit`` seconds or more XML than ``MAX_XML_BYTES`` end
    nmap (``_end``): ScanStopped, KeyboardInterrupt (raised again), NmapFailed. NmapFailed
    also when it doesn't start or ends with an error (``explain``)."""
    stop = stop or threading.Event()
    try:
        process = subprocess.Popen(list(command), stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   cwd=Path(command[0]).parent)
    except OSError as error:
        raise NmapFailed(f"nmap did not start ({type(error).__name__}).") from None
    assert process.stdout is not None and process.stderr is not None
    out = _Reader(process.stdout, MAX_XML_BYTES, _progress_lines(progress))
    err = _Reader(process.stderr, MAX_ERROR_BYTES)
    out.start()
    err.start()
    deadline = time.monotonic() + time_limit if time_limit else None
    try:
        while process.poll() is None:
            if stop.wait(POLL):
                raise ScanStopped()
            if out.overflow:
                raise NmapFailed(f"nmap wrote more than {MAX_XML_BYTES // 2**20} MB of XML; "
                                 "scan fewer ports.")
            if deadline is not None and time.monotonic() > deadline:
                raise NmapFailed(f"nmap was still running after {time_limit:.0f} s "
                                 "(TOOLKIT_SCAN_MAX_MIN); it was stopped.")
    except BaseException:  # the stop and the limits above, and Ctrl+C
        _end(process)
        raise
    finally:
        out.join(KILL_AFTER)
        err.join(KILL_AFTER)
    if out.overflow:
        raise NmapFailed(f"nmap wrote more than {MAX_XML_BYTES // 2**20} MB of XML; "
                         "scan fewer ports.")
    if process.returncode != 0:
        raise NmapFailed(explain(process.returncode, bytes(err.data)))
    return bytes(out.data)
