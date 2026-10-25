"""The toolkit's own tools as child processes (the menu runs each one this way)."""

import subprocess
import sys
from pathlib import Path


def run_toolkit(command: str, cwd: Path | None) -> int:
    """``python -m toolkit <command>`` with this Python, in ``cwd``; its exit code."""
    return subprocess.run([sys.executable, "-m", "toolkit", command], cwd=cwd,
                          check=False).returncode
