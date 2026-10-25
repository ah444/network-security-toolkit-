"""nmap, the optional scan engine (steps N2-N4): finding the program and building its command.

nmap is looked for only where the user or its installer put it: ``TOOLKIT_NMAP_PATH``, else
the folders on ``PATH`` given as full paths, else where its installer puts it (``PLACES``).
The current folder is never searched, so an ``nmap.exe`` planted there can't run, and nmap is
always started by its full path, from its own folder.

The command is built from an allow-list only: every option comes from the tables below, no
text the user typed reaches it, and the target is an address checked here. It always has
``-n -Pn -oX - --reason`` (``ALWAYS``): no DNS (the names were resolved before, after the
authorisation), no ping (the toolkit pings first), the XML on standard output, and why
nmap gave each state; and ``--stats-every 1s`` (``PROGRESS``) for the progress. Running it
is ``nmap_run``, reading its XML ``nmap_xml`` (step N3).

Service detection (``-sV``) also runs nmap's own "version" scripts, which no option turns
off (``VERSION_SCRIPTS_NOTE``, agreed with the user 2026-10-25); ``SCRIPTS`` is what
``--scripts`` adds.
"""

from __future__ import annotations

import ipaddress
import os
import platform
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from ..settings import nmap_path

MIN_VERSION = (7, 80)  # 2019: the --version-intensity, --reason and XML this toolkit reads
VERSION_PATTERN = re.compile(r"^Nmap version (\d+)\.(\d+)", re.MULTILINE)
VERSION_TIMEOUT = 15  # seconds for "nmap --version"

# Where nmap's installer puts it, after PATH: per OS, each a folder (the program is NAME in it).
PROGRAM_FILES = ("ProgramFiles(x86)", "ProgramFiles")  # Windows: the installer's "Nmap" folder
PLACES = {
    "darwin": ("/usr/local/bin", "/opt/homebrew/bin", "/opt/local/bin"),  # nmap.org, brew, ports
    "linux": ("/usr/bin", "/usr/local/bin", "/snap/bin"),
}
INSTALL = {
    "windows": "install it from https://nmap.org/download (the Windows installer, with Npcap)",
    "darwin": "install it with 'brew install nmap' or from https://nmap.org/download",
    "linux": "install your distribution's package (for example 'sudo apt install nmap')",
}

ALWAYS = ("-n", "-Pn", "-oX", "-", "--reason")
PROGRESS = ("--stats-every", "1s")  # a <taskprogress> element in the XML every second
INTENSITIES = {"light": 2, "normal": 7, "all": 9}  # --service-detection -> --version-intensity
TIMINGS = {"polite": "-T2", "normal": "-T3", "aggressive": "-T4"}  # --timing
# The only scripts that ever run, each talking to the target only: seven in nmap's "safe"
# and "discovery" categories, and http-server-header, a "version" script (it reads the
# Server header of the reply -sV gets). Never a category: nmap's "safe" has broadcast
# scripts that send to the whole network.
SCRIPTS = ("banner", "http-title", "http-server-header", "ssl-cert", "ssh-hostkey",
           "smb-protocols", "smb2-time", "rdp-ntlm-info")
SCRIPT_SETS = {"safe": SCRIPTS}  # --scripts
VERSION_SCRIPTS_NOTE = (
    "nmap's service detection also runs nmap's own version scripts (48 in nmap 7.992, such as "
    "rpcinfo and rpc-grind), each only on a service it recognised; nmap can't turn them off. "
    "One, iec61850-mms, is in nmap's 'intrusive' category: it talks to IEC 61850 industrial "
    "devices on port 102. --scripts safe adds only the 8 scripts the toolkit allows.")
PROTOCOLS = {"tcp": "T", "udp": "U"}
SCOPE_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}")  # an IPv6 zone: an interface name or number
NEEDS_PRIVILEGES = ("nmap's UDP scan and OS detection send raw packets: on Windows they need "
                    "Npcap (it comes with nmap's installer) and, if Npcap is set to "
                    "administrators only, an administrator's prompt; on Linux and macOS root. "
                    "Run the scan that way, or scan TCP only without OS detection.")
NPCAP_KEY = "\\".join(("SYSTEM", "CurrentControlSet", "Services", "npcap", "Parameters"))


class NmapMissing(Exception):
    """nmap can't be used here; the message says why and what to do."""


@dataclass(frozen=True)
class Nmap:
    """An nmap that can run: its full path and its version."""

    path: Path
    version: tuple[int, int]

    @property
    def version_text(self) -> str:
        """The version as nmap writes it, for example ``7.992``."""
        return f"{self.version[0]}.{self.version[1]}"


@dataclass(frozen=True)
class NmapOptions:
    """The nmap-only choices (the flags of step N4); each a key of its table."""

    service_detection: str = "normal"  # INTENSITIES
    timing: str = "normal"  # TIMINGS
    scripts: str = ""  # "" (none) or a key of SCRIPT_SETS
    os_detect: bool = False  # -O


def _system() -> str:
    """``windows``, ``darwin``, ``linux``, ... (``platform.system()`` in lower case)."""
    return platform.system().lower()


def program_name() -> str:
    """nmap's file name on this OS."""
    return "nmap.exe" if _system() == "windows" else "nmap"


def how_to_install() -> str:
    """What to do to get nmap on this OS (the Linux advice on any other Unix)."""
    return (f"To use it, {INSTALL.get(_system(), INSTALL['linux'])}, or set TOOLKIT_NMAP_PATH "
            "to its full path. The built-in engine (the default) needs nothing installed.")


def _variable(environ: Mapping[str, str], name: str) -> str:
    """``environ[name]``, '' when unset; on Windows in any case (``Path``, ``PATH``), as Windows
    reads it: a copy of ``os.environ`` there has the names in upper case."""
    if name in environ:
        return environ[name]
    if _system() == "windows":
        return next((value for key, value in environ.items() if key.upper() == name.upper()), "")
    return ""


def _usable(path: Path) -> bool:
    """``path`` is a file this user may run."""
    try:
        return path.is_file() and os.access(path, os.X_OK)
    except OSError:
        return False


def path_folders(environ: Mapping[str, str]) -> list[Path]:
    """The folders on ``PATH`` given as full paths, in order. An empty or relative entry
    (``.``) means the current folder, and so does an entry that is the current folder: all
    are left out."""
    here = Path.cwd().resolve()
    folders = []
    for entry in _variable(environ, "PATH").split(os.pathsep):
        folder = Path(entry.strip().strip('"'))
        if not entry.strip() or not folder.is_absolute():
            continue
        try:
            if folder.resolve() == here:
                continue
        except OSError:
            continue
        folders.append(folder)
    return folders


def install_places(environ: Mapping[str, str]) -> list[Path]:
    """The folders nmap's installer puts it in on this OS."""
    if _system() == "windows":
        folders = (Path(_variable(environ, name)) for name in PROGRAM_FILES)
        return [folder / "Nmap" for folder in folders if folder.is_absolute()]
    return [Path(place) for place in PLACES.get(_system(), PLACES["linux"])]


def locate(environ: Mapping[str, str] | None = None) -> Path:
    """nmap's full path: ``TOOLKIT_NMAP_PATH`` (the program or its folder; when set it is the
    only place looked at), else the first folder on ``PATH`` that has it, else its install
    folder. NmapMissing says where it was looked for and how to install it."""
    environ = os.environ if environ is None else environ
    try:
        chosen = nmap_path(environ)
    except ValueError as error:
        raise NmapMissing(f"{error}.") from None
    if chosen:
        path = Path(chosen)
        path = path / program_name() if path.is_dir() else path
        if not _usable(path):
            raise NmapMissing(f"TOOLKIT_NMAP_PATH is {chosen!r}, but there is no nmap program "
                              "there.")
        return path
    for folder in (*path_folders(environ), *install_places(environ)):
        if _usable(folder / program_name()):
            return folder / program_name()
    looked = ["TOOLKIT_NMAP_PATH", "the folders on PATH",
              *(str(place) for place in install_places(environ))]
    raise NmapMissing(f"nmap was not found (looked in {', '.join(looked)}). "
                      f"{how_to_install()}")


def read_version(output: str) -> tuple[int, int] | None:
    """(major, minor) from ``nmap --version``'s first line, ``Nmap version 7.992 (...)``."""
    match = VERSION_PATTERN.search(output)
    return (int(match.group(1)), int(match.group(2))) if match else None


def check_version(path: Path) -> tuple[int, int]:
    """Runs ``path --version`` (no shell, from nmap's own folder, nothing sent anywhere) and
    gives its version; NmapMissing if it doesn't run, doesn't say, or is older than
    ``MIN_VERSION``."""
    try:
        done = subprocess.run([str(path), "--version"], capture_output=True, check=False,
                              stdin=subprocess.DEVNULL, cwd=path.parent,
                              timeout=VERSION_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise NmapMissing(f"{path} did not run ({type(error).__name__}). "
                          f"{how_to_install()}") from None
    version = read_version(done.stdout.decode("utf-8", "replace"))
    if version is None:
        raise NmapMissing(f"{path} did not say which nmap version it is; is it nmap? "
                          f"{how_to_install()}")
    if version < MIN_VERSION:
        raise NmapMissing(f"nmap {version[0]}.{version[1]} ({path}) is too old: "
                          f"{MIN_VERSION[0]}.{MIN_VERSION[1]} or newer is needed. "
                          f"{how_to_install()}")
    return version


def find_nmap(environ: Mapping[str, str] | None = None) -> Nmap:
    """The nmap to use (``locate``, then ``check_version``); NmapMissing says why there is none."""
    path = locate(environ)
    return Nmap(path, check_version(path))


def _is_admin() -> bool:
    """This process runs as an administrator (Windows) or root (other OSes)."""
    if _system() == "windows":
        import ctypes
        windll = getattr(ctypes, "windll", None)  # only Windows' ctypes has it
        try:
            return bool(windll.shell32.IsUserAnAdmin()) if windll else False
        except (AttributeError, OSError):
            return False
    return getattr(os, "geteuid", lambda: -1)() == 0


def _npcap_admin_only() -> bool:
    """Npcap is set to work for administrators only (its ``AdminOnly`` value; read only).
    True when it can't be read, so a doubt means no raw packets."""
    if sys.platform != "win32":
        return True
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, NPCAP_KEY) as key:
            return bool(winreg.QueryValueEx(key, "AdminOnly")[0])
    except OSError:
        return True


def has_privileges(environ: Mapping[str, str] | None = None) -> bool:
    """nmap can send raw packets here (``-sS``, ``-sU``, ``-O``): on Windows Npcap is installed
    (``System32/Npcap/wpcap.dll``) and works for this user (not administrators only, or this
    is an administrator); on Linux and macOS this is root."""
    if _system() != "windows":
        return _is_admin()
    environ = os.environ if environ is None else environ
    root = Path(_variable(environ, "SystemRoot"))
    if not root.is_absolute() or not (root / "System32" / "Npcap" / "wpcap.dll").is_file():
        return False
    return not _npcap_admin_only() or _is_admin()


def target_address(host: str) -> str:
    """``host`` as nmap gets it: an IP address only (no name, network or option), an IPv6
    zone only if it is a plain interface name or number. ValueError for anything else."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(f"nmap gets an IP address only, not {host!r}") from None
    scope = getattr(address, "scope_id", None)
    if scope is not None and not SCOPE_PATTERN.fullmatch(scope):
        raise ValueError(f"nmap gets an IP address only, not {host!r}")
    return str(address)


def port_ranges(ports: Iterable[int]) -> str:
    """The ports, sorted and each once, as nmap's list: ``22,80-82,443``. ValueError for an
    empty list or anything that isn't a port number 1-65535."""
    given = list(ports)
    if not given:
        raise ValueError("nmap needs at least one port")
    if any(type(n) is not int or not 1 <= n <= 65535 for n in given):
        raise ValueError("nmap gets port numbers from 1 to 65535 only")
    numbers = sorted(set(given))
    ranges, start = [], numbers[0]
    for before, after in pairwise([*numbers, 0]):  # 0: no port follows the last one
        if after != before + 1:
            ranges.append(str(start) if start == before else f"{start}-{before}")
            start = after
    return ",".join(ranges)


def _choice(table: Mapping[str, object], value: str, what: str) -> None:
    """ValueError naming ``what`` and the allowed values when ``value`` isn't in ``table``."""
    if value not in table:
        raise ValueError(f"Unknown {what} {value!r}; use one of {', '.join(table)}.")


def nmap_command(nmap: Path, host: str, ports: Iterable[int], protocols: Sequence[str],
                 options: NmapOptions, *, privileged: bool, max_minutes: int) -> list[str]:
    """The arguments to run nmap with (no shell): ``ALWAYS``, ``PROGRESS``, ``-6`` for IPv6,
    the scan types (TCP: ``-sS`` with privileges, else ``-sT --unprivileged``; UDP: ``-sU``),
    ``-sV`` at the chosen intensity, the timing, ``-O`` and the scripts if chosen,
    ``--host-timeout`` (``max_minutes``), the ports as ``T:``/``U:`` lists and the address last.

    ValueError for an unknown choice, a bad address or port, or UDP or OS detection without
    privileges (``NEEDS_PRIVILEGES``)."""
    address = target_address(host)
    _choice(INTENSITIES, options.service_detection, "service detection")
    _choice(TIMINGS, options.timing, "timing")
    if options.scripts:
        _choice(SCRIPT_SETS, options.scripts, "script set")
    if not protocols or any(p not in PROTOCOLS for p in protocols):
        raise ValueError(f"nmap scans {' and '.join(PROTOCOLS)} only, not {list(protocols)!r}")
    if type(max_minutes) is not int or max_minutes < 1:
        raise ValueError("nmap's host timeout must be a whole number of minutes >= 1")
    if not privileged and ("udp" in protocols or options.os_detect):
        raise ValueError(NEEDS_PRIVILEGES)
    port_list = port_ranges(ports)
    command = [str(nmap), *ALWAYS, *PROGRESS]
    if ipaddress.ip_address(address).version == 6:
        command.append("-6")
    if "tcp" in protocols:
        command += ["-sS"] if privileged else ["-sT", "--unprivileged"]
    if "udp" in protocols:
        command.append("-sU")
    command += ["-sV", "--version-intensity", str(INTENSITIES[options.service_detection]),
                TIMINGS[options.timing]]
    if options.os_detect:
        command.append("-O")
    if options.scripts:
        command += ["--script", ",".join(SCRIPT_SETS[options.scripts])]
    command += ["--host-timeout", f"{max_minutes}m",
                "-p", ",".join(f"{PROTOCOLS[p]}:{port_list}" for p in PROTOCOLS
                               if p in protocols),
                address]
    return command
