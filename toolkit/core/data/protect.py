"""Who may open this install's private data (v5.3 step HS5, RD1): the ``instance/`` folder (the
signing key, the runs, the bundles, the logs) and ``.env`` (the API keys).

RD1 trusts this computer as one domain; other accounts on it are kept out by the operating
system's permissions only. On Windows a folder on a drive other than the system's often
inherits "Everyone: Full control" from the drive's root, so ``protect`` sets the permissions
itself: nothing inherited from above, full control for the current user, SYSTEM (backups and
services) and Administrators only; every file made in the folder inherits that. It uses
``icacls`` and ``whoami``, both part of Windows. On Linux and macOS it is ``chmod``: 700 for a
folder, 600 for a file.

A path that can't be protected (a FAT drive has no permissions, a network drive may refuse)
is never an error: ``protect`` says why and the toolkit goes on. Each path is protected once
per process.
"""

from __future__ import annotations

import functools
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from .settings import env_file, instance_dir

SYSTEM, ADMINISTRATORS = "S-1-5-18", "S-1-5-32-544"
SDDL_NAMES = {"SY": SYSTEM, "BA": ADMINISTRATORS, "WD": "S-1-1-0", "AU": "S-1-5-11",
              "BU": "S-1-5-32-545"}  # how the ACL text names them
NAMES = {"S-1-1-0": "Everyone", "S-1-5-11": "Authenticated Users", "S-1-5-32-545": "Users"}
_ACE = re.compile(r"\((A|D);([A-Z]*);[^;]*;[^;]*;[^;]*;([^)]+)\)")  # (type;flags;...;SID)
_done: dict[Path, str | None] = {}


def protect(path: str | Path) -> str | None:
    """Limits who may open ``path`` (a folder or a file), once in this process; why it
    couldn't, or None."""
    path = Path(path).absolute()
    if path not in _done:
        try:
            _apply(path)
            _done[path] = None
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            _done[path] = (f"Could not limit who may open {path} ({error}): other accounts "
                           "on this computer may be able to read it.")
    return _done[path]


def protect_private_data(environ: Mapping[str, str] | None = None) -> list[str]:
    """``instance/`` (made now if missing, so what is made in it later inherits its
    permissions) and ``.env`` when there is one; the problems, if any."""
    folder = instance_dir(environ)
    folder.mkdir(parents=True, exist_ok=True)
    env = env_file(environ)
    return [problem for path in (folder, *([env] if env.is_file() else []))
            if (problem := protect(path)) is not None]


def _apply(path: Path) -> None:
    """The permissions of ``path`` set, then checked on Windows (ValueError if another
    account can still open it)."""
    if sys.platform != "win32":
        os.chmod(path, 0o700 if path.is_dir() else 0o600)
        return
    owners = (current_user_sid(), SYSTEM, ADMINISTRATORS)
    inherit = "(OI)(CI)" if path.is_dir() else ""  # a folder's files and folders inherit it
    _run(["icacls", str(path), "/inheritance:r", "/grant:r",
          *(f"*{sid}:{inherit}F" for sid in owners)])
    inherited, others = access(path, owners)  # what Windows says now, not what was asked
    if inherited:
        raise ValueError("it still inherits permissions from the folder above")
    if others:
        raise ValueError("still open to " + ", ".join(NAMES.get(sid, sid) for sid in others))


def access(path: Path, owners: tuple[str, ...]) -> tuple[bool, list[str]]:
    """Windows: whether ``path`` inherits permissions from above, and the accounts (SIDs)
    other than ``owners`` it allows."""
    handle, saved = tempfile.mkstemp(suffix=".acl")
    os.close(handle)
    try:
        _run(["icacls", str(path), "/save", saved])
        text = Path(saved).read_bytes().decode("utf-16-le")
    finally:
        os.unlink(saved)
    acl = text.splitlines()[1] if len(text.splitlines()) > 1 else ""
    dacl = acl[acl.find("D:"):]
    aces = _ACE.findall(dacl)
    inherited = "P" not in dacl[2:dacl.find("(")] or any("ID" in flags for _, flags, _ in aces)
    allowed = {SDDL_NAMES.get(sid, sid) for kind, _, sid in aces if kind == "A"}
    return inherited, sorted(allowed - set(owners))


@functools.cache
def current_user_sid() -> str:
    """The SID of the account this process runs as (``whoami /user``)."""
    out = _run(["whoami", "/user", "/fo", "csv", "/nh"])
    sid = out.strip().rsplit(",", 1)[-1].strip('"')
    if not sid.startswith("S-1-"):
        raise ValueError(f"whoami gave no SID ({out.strip()!r})")
    return sid


def _run(command: list[str]) -> str:
    """A Windows command's output; OSError with its message when it fails."""
    done = subprocess.run(command, capture_output=True, text=True, errors="replace",
                          timeout=30, check=False)
    if done.returncode != 0:
        message = (done.stderr or done.stdout).strip().splitlines()
        raise OSError(f"{command[0]}: {message[0] if message else f'exit code {done.returncode}'}")
    return done.stdout
