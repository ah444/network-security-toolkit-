"""Step N2: nmap is found only where the user or its installer put it (never in the current
folder), its version is checked, and its command is built from an allow-list only."""

import itertools
import subprocess
from pathlib import Path

import pytest

from toolkit.core.data import settings
from toolkit.core.data.recon import nmap
from toolkit.core.data.recon.nmap import NmapMissing, NmapOptions

NMAP = Path("/opt/nmap/nmap")
DEFAULTS = NmapOptions()
FORBIDDEN = ("--script-args", "--script-args-file", "-D", "-S", "-f", "-sI", "--data",
             "--data-string", "--data-length", "-iL", "-oN", "-oG", "-oA", "--resume",
             "--datadir", "-e", "--spoof-mac", "--proxies", "-sC", "-A")


@pytest.fixture
def on(monkeypatch):
    """Pretend to be on another OS: ``on("windows")``."""
    return lambda system: monkeypatch.setattr(nmap.platform, "system", lambda: system.title())


def _program(folder: Path, name: str = "nmap") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"")
    path.chmod(0o755)
    return path


# Finding it

def test_toolkit_nmap_path_is_the_only_place_when_set(tmp_path, on):
    on("linux")
    chosen = _program(tmp_path / "mine")
    _program(tmp_path / "onpath")
    environ = {"TOOLKIT_NMAP_PATH": str(chosen), "PATH": str(tmp_path / "onpath")}
    assert nmap.locate(environ) == chosen
    assert nmap.locate({"TOOLKIT_NMAP_PATH": f'"{tmp_path / "mine"}"'}) == chosen, \
        "its folder, in quotes, works too"
    with pytest.raises(NmapMissing, match="TOOLKIT_NMAP_PATH is .*no nmap program there"):
        nmap.locate({**environ, "TOOLKIT_NMAP_PATH": str(tmp_path / "nothing")})


def test_a_relative_toolkit_nmap_path_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _program(tmp_path)
    for relative in ("nmap", "./nmap", "bin/nmap"):
        with pytest.raises(NmapMissing, match="TOOLKIT_NMAP_PATH must be nmap's full path"):
            nmap.locate({"TOOLKIT_NMAP_PATH": relative})
        with pytest.raises(ValueError, match="TOOLKIT_NMAP_PATH must be nmap's full path"):
            settings.nmap_path({"TOOLKIT_NMAP_PATH": relative})
    assert settings.check_settings("scan", {"TOOLKIT_NMAP_PATH": "nmap"}) == [
        "TOOLKIT_NMAP_PATH must be nmap's full path, not 'nmap'"]
    assert settings.check_settings("scan", {"TOOLKIT_NMAP_PATH": str(tmp_path)}) == []


def test_the_first_folder_on_path_that_has_it(tmp_path, on):
    on("linux")
    first = _program(tmp_path / "b")
    _program(tmp_path / "c")
    path = str(tmp_path / "a") + nmap.os.pathsep + str(tmp_path / "b") + nmap.os.pathsep + str(
        tmp_path / "c")
    assert nmap.locate({"PATH": path}) == first


def test_the_current_folder_is_never_searched(tmp_path, monkeypatch, on):
    """A planted nmap in the current folder: PATH's empty entry, '.', a relative folder and
    the current folder by its full path all mean it; none is looked in."""
    on("linux")
    monkeypatch.chdir(tmp_path)
    _program(tmp_path)
    _program(tmp_path / "bin")
    monkeypatch.setattr(nmap, "PLACES", {"linux": ()})
    sep = nmap.os.pathsep
    for path in ("", sep, ".", "bin", f".{sep}bin{sep}", str(tmp_path), f'"{tmp_path}"'):
        assert nmap.path_folders({"PATH": path}) == []
        with pytest.raises(NmapMissing, match="nmap was not found"):
            nmap.locate({"PATH": path})


def test_the_install_folder_after_path(tmp_path, on):
    on("windows")
    installed = _program(tmp_path / "x86" / "Nmap", "nmap.exe")
    environ = {"ProgramFiles(x86)": str(tmp_path / "x86"), "ProgramFiles": str(tmp_path / "pf"),
               "PATH": str(tmp_path / "empty")}
    assert nmap.install_places(environ) == [tmp_path / "x86" / "Nmap", tmp_path / "pf" / "Nmap"]
    assert nmap.locate(environ) == installed
    assert nmap.install_places({"ProgramFiles": "relative"}) == []
    same = {"ProgramFiles(x86)": str(tmp_path / "pf"), "ProgramFiles": str(tmp_path / "pf")}
    assert nmap.install_places(same) == [tmp_path / "pf" / "Nmap"]  # the same folder: once
    upper = {"PROGRAMFILES(X86)": str(tmp_path / "x86"), "Path": str(tmp_path / "empty")}
    assert nmap.locate(upper) == installed, "a copy of os.environ: the names in upper case"
    assert nmap.path_folders(upper) == [tmp_path / "empty"]
    on("linux")
    assert nmap.path_folders({"Path": str(tmp_path)}) == [], "case counts on Linux and macOS"
    on("linux")
    assert nmap.install_places({}) == [Path("/usr/bin"), Path("/usr/local/bin"),
                                       Path("/snap/bin")]
    on("darwin")
    assert Path("/opt/homebrew/bin") in nmap.install_places({})


@pytest.mark.parametrize(("system", "advice"), [
    ("windows", "https://nmap.org/download (the Windows installer, with Npcap)"),
    ("darwin", "brew install nmap"),
    ("linux", "sudo apt install nmap"),
    ("freebsd", "sudo apt install nmap"),
])
def test_not_found_says_how_to_install(tmp_path, monkeypatch, on, system, advice):
    on(system)
    monkeypatch.setattr(nmap, "PLACES", {"linux": (), "darwin": ()})
    with pytest.raises(NmapMissing) as raised:
        nmap.locate({"PATH": str(tmp_path)})
    message = str(raised.value)
    assert advice in message and "TOOLKIT_NMAP_PATH" in message
    assert "The built-in engine (the default) needs nothing installed." in message


# Its version

@pytest.mark.parametrize(("output", "version"), [
    ("Nmap version 7.992 ( https://nmap.org )\nPlatform: i686-pc-windows-windows\n", (7, 992)),
    ("Nmap version 7.80 ( https://nmap.org )\n", (7, 80)),
    ("Nmap version 7.94SVN ( https://nmap.org )\n", (7, 94)),
    ("\nNmap version 7.95 ( https://nmap.org )\n", (7, 95)),
    ("Ncat: Version 7.92 ( https://nmap.org/ncat )\n", None),
    ("not nmap\n", None),
    ("", None),
])
def test_read_version(output, version):
    assert nmap.read_version(output) == version


def test_the_version_is_asked_safely(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, b"Nmap version 7.992 ( https://nmap.org )\n")

    monkeypatch.setattr(nmap.subprocess, "run", run)
    assert nmap.check_version(NMAP) == (7, 992)
    ((args, kwargs),) = calls
    assert args == [str(NMAP), "--version"], "its full path, no shell"
    assert kwargs["cwd"] == NMAP.parent and kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["timeout"] and not kwargs.get("shell")


@pytest.mark.parametrize(("reply", "message"), [
    (b"Nmap version 7.70 ( https://nmap.org )\n", r"nmap 7\.70 .* is too old: 7\.80 or newer"),
    (b"Nmap version 7.8 ( https://nmap.org )\n", r"nmap 7\.8 .* is too old"),
    (b"hello\n", "did not say which nmap version it is"),
    (OSError("no"), r"did not run \(OSError\)"),
    (subprocess.TimeoutExpired("nmap", 15), r"did not run \(TimeoutExpired\)"),
])
def test_a_version_it_cant_use(monkeypatch, reply, message):
    def run(args, **kwargs):
        if isinstance(reply, BaseException):
            raise reply
        return subprocess.CompletedProcess(args, 0, reply)

    monkeypatch.setattr(nmap.subprocess, "run", run)
    with pytest.raises(NmapMissing, match=message):
        nmap.check_version(NMAP)


def test_find_nmap(tmp_path, monkeypatch, on):
    on("linux")
    path = _program(tmp_path)
    monkeypatch.setattr(nmap, "check_version", lambda found: (7, 992))
    found = nmap.find_nmap({"TOOLKIT_NMAP_PATH": str(path)})
    assert (found.path, found.version, found.version_text) == (path, (7, 992), "7.992")


# Its command

def _command(host="192.168.56.10", ports=(22, 80), protocols=("tcp",), options=DEFAULTS,
             privileged=False, minutes=60):
    return nmap.nmap_command(NMAP, host, ports, protocols, options, privileged=privileged,
                             max_minutes=minutes)


def test_the_default_command():
    assert _command() == [str(NMAP), "-n", "-Pn", "-oX", "-", "--reason", "--stats-every", "1s",
                          "-sT",
                          "--unprivileged", "-sV", "--version-intensity", "7", "-T3",
                          "--host-timeout", "60m", "-p", "T:22,80", "192.168.56.10"]


def test_every_choice():
    options = NmapOptions(service_detection="all", timing="aggressive", scripts="safe",
                          os_detect=True)
    assert _command("::1", [53, 22, 23, 24, 80], ("tcp", "udp"), options, privileged=True,
                    minutes=5) == [
        str(NMAP), "-n", "-Pn", "-oX", "-", "--reason", "--stats-every", "1s", "-6", "-sS",
        "-sU", "-sV",
        "--version-intensity", "9", "-T4", "-O", "--script",
        "banner,http-title,http-server-header,ssl-cert,ssh-hostkey,smb-protocols,smb2-time,"
        "rdp-ntlm-info", "--host-timeout", "5m", "-p", "T:22-24,53,80,U:22-24,53,80", "::1"]
    assert _command(protocols=("udp",), privileged=True)[-3:] == ["-p", "U:22,80",
                                                                   "192.168.56.10"]


@pytest.mark.parametrize(("choice", "flags"), [
    ({"service_detection": "light"}, ["--version-intensity", "2"]),
    ({"service_detection": "normal"}, ["--version-intensity", "7"]),
    ({"timing": "polite"}, ["-T2"]),
    ({"timing": "normal"}, ["-T3"]),
])
def test_each_choice_is_its_flag(choice, flags):
    command = _command(options=NmapOptions(**choice))
    start = command.index(flags[0])
    assert command[start:start + len(flags)] == flags


@pytest.mark.parametrize(("ports", "listed"), [
    ([80], "80"), ([1, 2, 3], "1-3"), ([3, 1, 2, 2], "1-3"), ([22, 80, 81, 82, 443], "22,80-82,443"),
    (range(1, 1025), "1-1024"), ([1, 3, 5], "1,3,5"), ([65535, 1], "1,65535"),
])
def test_port_ranges(ports, listed):
    assert nmap.port_ranges(ports) == listed


@pytest.mark.parametrize("ports", [[], [0], [65536], [-1], ["80"], [80.0], [True], ["1-2,3"]])
def test_bad_ports_are_refused(ports):
    with pytest.raises(ValueError, match="nmap"):
        nmap.port_ranges(ports)


@pytest.mark.parametrize("host", [
    "scanme.nmap.org", "localhost", "192.168.1.0/24", "192.168.1.1-5", "-oN x", "--script=vuln",
    "1.2.3.4 -sC", "1.2.3.4,5.6.7.8", "", " 1.2.3.4", "fe80::1%eth0 -sC", "fe80::1%a/b",
])
def test_only_an_address_is_a_target(host):
    with pytest.raises(ValueError, match="nmap gets an IP address only"):
        _command(host)


def test_addresses_as_nmap_gets_them():
    assert _command("fe80::1%eth0")[-1] == "fe80::1%eth0"
    assert _command("FE80:0::1")[-1] == "fe80::1"
    assert _command("10.0.0.1")[-1] == "10.0.0.1" and "-6" not in _command("10.0.0.1")


@pytest.mark.parametrize(("protocols", "options"), [
    (("udp",), NmapOptions()),
    (("tcp", "udp"), NmapOptions()),
    (("tcp",), NmapOptions(os_detect=True)),
])
def test_udp_and_os_detection_need_privileges(protocols, options):
    with pytest.raises(ValueError, match="send raw packets: on Windows they need Npcap"):
        _command(protocols=protocols, options=options)
    assert _command(protocols=protocols, options=options, privileged=True)


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"options": NmapOptions(service_detection="max")}, "Unknown service detection 'max'"),
    ({"options": NmapOptions(timing="insane")}, "Unknown timing 'insane'; use one of polite"),
    ({"options": NmapOptions(scripts="vuln")}, "Unknown script set 'vuln'; use one of safe"),
    ({"options": NmapOptions(scripts="default")}, "Unknown script set"),
    ({"protocols": ("sctp",)}, "nmap scans tcp and udp only"),
    ({"protocols": ()}, "nmap scans tcp and udp only"),
    ({"minutes": 0}, "host timeout"),
    ({"minutes": "60"}, "host timeout"),
])
def test_unknown_choices_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _command(**kwargs)


def test_no_choice_ever_gives_a_forbidden_option():
    """Every combination: the fixed options are there, the address is last, the scripts are
    the allow-listed ones only, and no option outside the allow-list appears."""
    allowed = {*nmap.ALWAYS, *nmap.PROGRESS, "-6", "-sS", "-sT", "--unprivileged", "-sU", "-sV",
               "--version-intensity", *map(str, nmap.INTENSITIES.values()),
               *nmap.TIMINGS.values(), "-O", "--script", ",".join(nmap.SCRIPTS),
               "--host-timeout", "-p"}
    count = 0
    for host, protocols, intensity, timing, scripts, os_detect, privileged in itertools.product(
            ("10.0.0.5", "fe80::5"), (("tcp",), ("udp",), ("tcp", "udp")), nmap.INTENSITIES,
            nmap.TIMINGS, ("", *nmap.SCRIPT_SETS), (False, True), (False, True)):
        options = NmapOptions(intensity, timing, scripts, os_detect)
        try:
            command = _command(host, [22, 80], protocols, options, privileged, 30)
        except ValueError:
            assert not privileged and ("udp" in protocols or os_detect)
            continue
        count += 1
        assert command[0] == str(NMAP) and command[1:8] == [*nmap.ALWAYS, *nmap.PROGRESS]
        assert command[-1] == host and command[-4:-1] == ["30m", "-p", command[-2]]
        rest = command[1:-4]
        assert set(rest) <= allowed, set(rest) - allowed
        assert not set(command) & set(FORBIDDEN)
        assert not any(part.startswith(("--script-", "--data")) for part in command)
        if scripts:
            assert command[command.index("--script") + 1].split(",") == list(nmap.SCRIPTS)
    assert count == 216 + 36, "all 216 with privileges, the 36 TCP ones without -O without"


def test_the_scripts_are_the_agreed_ones():
    assert nmap.SCRIPTS == ("banner", "http-title", "http-server-header", "ssl-cert",
                            "ssh-hostkey", "smb-protocols", "smb2-time", "rdp-ntlm-info")
    assert set(nmap.SCRIPT_SETS) == {"safe"}


def test_the_time_limit_setting():
    assert settings.scan_limits({}).max_minutes == 60
    assert settings.scan_limits({"TOOLKIT_SCAN_MAX_MIN": "5"}).max_minutes == 5
    with pytest.raises(ValueError, match="^TOOLKIT_SCAN_MAX_MIN must be a whole number >= 1"):
        settings.scan_limits({"TOOLKIT_SCAN_MAX_MIN": "0"})
    assert settings.check_settings("scan", {"TOOLKIT_SCAN_MAX_MIN": "abc"}) == [
        "TOOLKIT_SCAN_MAX_MIN must be a whole number >= 1, not 'abc'"]


# Step N3: whether nmap can send raw packets

@pytest.fixture
def windows(tmp_path, on, monkeypatch):
    """Windows with its folder in ``tmp_path``: ``npcap=True`` puts Npcap's wpcap.dll there."""
    on("windows")

    def make(npcap=True, admin_only=False, admin=False):
        if npcap:
            (tmp_path / "System32" / "Npcap").mkdir(parents=True, exist_ok=True)
            (tmp_path / "System32" / "Npcap" / "wpcap.dll").write_bytes(b"")
        monkeypatch.setattr(nmap, "_npcap_admin_only", lambda: admin_only)
        monkeypatch.setattr(nmap, "_is_admin", lambda: admin)
        return {"SystemRoot": str(tmp_path)}
    return make


@pytest.mark.parametrize(("npcap", "admin_only", "admin", "privileged"), [
    (False, False, True, False),  # no Npcap: no raw packets, even for an administrator
    (True, False, False, True),  # Npcap for every user (as on this PC)
    (True, True, False, False),  # Npcap for administrators only
    (True, True, True, True),
])
def test_privileges_on_windows(windows, npcap, admin_only, admin, privileged):
    assert nmap.has_privileges(windows(npcap, admin_only, admin)) is privileged


def test_privileges_without_systemroot(windows):
    windows()
    assert nmap.has_privileges({"SystemRoot": ""}) is False


@pytest.mark.parametrize("root", [True, False])
def test_privileges_on_linux_and_macos_are_root(on, monkeypatch, root):
    monkeypatch.setattr(nmap, "_is_admin", lambda: root)
    for system in ("linux", "darwin"):
        on(system)
        assert nmap.has_privileges({}) is root


def test_npcap_admin_only_is_read_only_on_windows(monkeypatch):
    monkeypatch.setattr(nmap.sys, "platform", "linux")
    assert nmap._npcap_admin_only() is True, "elsewhere: a doubt is no raw packets"
