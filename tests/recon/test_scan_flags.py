"""Step 1.4c: ``scan TARGET [flags]`` asks nothing (but the authorization question without
``--i-am-authorized``), sends nothing online unless a flag says so; Ctrl+C cancels cleanly.
No test reaches the network: ping, the port checks and the lookups are fakes."""

import builtins
import json
import threading
import time

import pytest

from toolkit.core.data.recon import scanner as recon_scanner
from toolkit.core.services.recon import targets
from toolkit.core.data.recon.scanner import PortResult

AUTHORIZED = "--i-am-authorized"


class FakeIntel:
    """Records what would leave the computer; finds no CVEs."""

    made = []

    def __init__(self, keys, use_ai=False, stop=None):
        self.keys, self.use_ai, self.stop = keys, use_ai, stop
        self.looked_up, self.abuse_checked = [], []
        FakeIntel.made.append(self)

    def lookup(self, software):
        self.looked_up.append(software)
        return {"cves": [], "cve_source": "", "max_cvss": None, "severity": "UNKNOWN",
                "severity_unconfirmed": False, "ai_explanation": "", "errors": []}

    def lookup_many(self, products):
        return {product: self.lookup(product) for product in products}

    def abuse_report(self, ip):
        self.abuse_checked.append(ip)
        return {"ipAddress": ip, "abuseConfidenceScore": 0}, None

    def sent(self):
        return {"NVD": len(self.looked_up), "AI (OpenRouter)": 0,
                "AbuseIPDB": len(self.abuse_checked)}


@pytest.fixture
def fake(fake_ports, scan_service, scanner, monkeypatch, tmp_path):
    """The scanner with fake ping, port checks and lookups; every question is recorded
    and fails the test unless it is the authorization one."""
    calls = {"scans": [], "prompts": [], "alive": True, "authorize": "yes",
             "open": [PortResult(22, "tcp", "open", "ssh", "SSH-2.0-OpenSSH_8.9p1")]}
    FakeIntel.made = []

    def fake_input(prompt=""):
        calls["prompts"].append(prompt)
        if "authorised to scan" in prompt:
            answer = calls["authorize"]
            if isinstance(answer, type):
                raise answer
            return answer
        raise AssertionError(f"asked {prompt!r}")

    def fake_scan(host, ports, protocols, progress=None):
        calls["scans"].append((host, list(ports), list(protocols), progress))
        return list(calls["open"])

    monkeypatch.setattr(builtins, "input", fake_input)
    monkeypatch.setattr(scan_service, "ping_host", lambda host, timeout=1: (calls["alive"], 64))
    fake_ports(fake_scan)
    monkeypatch.setattr(scan_service, "Intel", FakeIntel)
    monkeypatch.setattr(scanner, "load_api_keys",
                        lambda: {"openrouter": "", "abuseipdb": "", "nvd": ""})
    monkeypatch.setattr("toolkit.core.data.recon.scan_files.scans_dir", lambda: tmp_path / "scans")
    calls["scans_dir"] = tmp_path / "scans"
    return calls


def _keys(scanner, monkeypatch, **keys):
    values = {"openrouter": "", "abuseipdb": "", "nvd": "", **keys}
    monkeypatch.setattr(scanner, "load_api_keys", lambda: values)


def test_parse_ports():
    assert targets.parse_ports("80") == [80]
    assert targets.parse_ports(" 22, 80,1-3,80 ") == [1, 2, 3, 22, 80]
    assert targets.parse_ports("1-1024") == list(range(1, 1025))
    assert targets.parse_ports("65535") == [65535]
    for spec, words in (("", "No ports given"), (" , ", "No ports given"),
                        ("1,,2", "empty item"), ("0", "between 1 and 65535"),
                        ("65536", "between 1 and 65535"), ("9-3", "after the end port"),
                        ("http", "whole numbers"), ("-5", "whole numbers"),
                        ("1-", "whole numbers"), ("1-2-3", "whole numbers")):
        with pytest.raises(ValueError, match=words):
            targets.parse_ports(spec)


def test_target_with_flags_asks_nothing(scan_screen, scanner, fake, capsys):
    assert scanner.main(["127.0.0.1", "-p", "22,80", "--protocol", "BOTH", "-v",
                         AUTHORIZED]) == 0
    assert fake["prompts"] == []
    [(host, ports, protocols, progress)] = fake["scans"]
    assert (host, ports, protocols) == ("127.0.0.1", [22, 80], ["tcp", "udp"])
    assert isinstance(progress, scan_screen.ConsoleProgress)
    out = capsys.readouterr().out
    assert "Total open ports found: 1" in out
    assert "export" not in out.lower(), "no export and no export question without --export"


def test_defaults_nothing_sent(scanner, fake, monkeypatch):
    """Defaults: ports 1-1024, TCP, no progress, no AI, no AbuseIPDB, even with keys."""
    _keys(scanner, monkeypatch, openrouter="or-key", abuseipdb="ab-key")
    assert scanner.main(["8.8.8.8", AUTHORIZED]) == 0
    [(host, ports, protocols, progress)] = fake["scans"]
    assert (ports, protocols, progress) == (list(range(1, 1025)), ["tcp"], None)
    [intel] = FakeIntel.made
    assert intel.use_ai is False
    assert intel.abuse_checked == []
    assert intel.looked_up == ["OpenSSH 8.9"], "CVE lookups (NVD) stay on"


def test_authorization_question_stays_without_the_flag(scanner, fake, capsys):
    assert scanner.main(["127.0.0.1", "-p", "22"]) == 0
    assert len(fake["prompts"]) == 1 and "authorised to scan 127.0.0.1" in fake["prompts"][0]
    fake["authorize"] = "no"
    fake["scans"].clear()
    assert scanner.main(["127.0.0.1", "-p", "22"]) == 2
    assert fake["scans"] == [], "no packet without the authorization"
    assert "Scan cancelled: only scan systems you are authorised" in capsys.readouterr().out


def test_ai_flag(scanner, fake, monkeypatch, capsys):
    assert scanner.main(["127.0.0.1", "-p", "22", "--ai", AUTHORIZED]) == 0
    assert FakeIntel.made[-1].use_ai is False
    assert "--ai: no OPENROUTER_API_KEY set, so no AI explanation." in capsys.readouterr().out
    assert len(fake["scans"]) == 1, "the scan goes on without the key"
    _keys(scanner, monkeypatch, openrouter="or-key")
    assert scanner.main(["127.0.0.1", "-p", "22", "--ai", AUTHORIZED]) == 0
    assert FakeIntel.made[-1].use_ai is True
    assert "never the raw banner" in capsys.readouterr().out


def test_abuseipdb_flag(scanner, fake, monkeypatch, capsys):
    assert scanner.main(["8.8.8.8", "-p", "22", "--abuseipdb", AUTHORIZED]) == 0
    assert FakeIntel.made[-1].abuse_checked == []
    assert "--abuseipdb: no ABUSEIPDB_API_KEY set" in capsys.readouterr().out
    assert len(fake["scans"]) == 1
    _keys(scanner, monkeypatch, abuseipdb="ab-key")
    assert scanner.main(["8.8.8.8", "-p", "22", "--abuseipdb", AUTHORIZED]) == 0
    assert FakeIntel.made[-1].abuse_checked == ["8.8.8.8"]
    assert "AbuseIPDB reputation info for 8.8.8.8" in capsys.readouterr().out
    assert scanner.main(["10.0.0.1", "-p", "22", "--abuseipdb", AUTHORIZED]) == 0
    assert FakeIntel.made[-1].abuse_checked == [], "only public IPs are sent"
    assert fake["prompts"] == []


def test_unresponsive_host(scanner, fake, capsys):
    fake["alive"] = False
    assert scanner.main(["10.0.0.9", "-p", "22", AUTHORIZED]) == 0
    assert fake["scans"] == []
    assert "Skipped (--scan-unresponsive scans it anyway)." in capsys.readouterr().out
    assert scanner.main(["10.0.0.9", "-p", "22", "--scan-unresponsive", AUTHORIZED]) == 0
    assert [scan[0] for scan in fake["scans"]] == ["10.0.0.9"]
    assert fake["prompts"] == []


def test_export_flag(scanner, fake, capsys):
    assert scanner.main(["127.0.0.1", "-p", "22", "--export", "JSON", AUTHORIZED]) == 0
    [path] = fake["scans_dir"].glob("scan-*.json")
    assert json.loads(path.read_text(encoding="utf-8"))[0]["port"] == 22
    assert f"Results exported to {path}" in capsys.readouterr().out
    fake["open"] = []
    assert scanner.main(["127.0.0.1", "-p", "22", "--export", "csv", AUTHORIZED]) == 0
    assert "No open port results to export." in capsys.readouterr().out
    assert list(fake["scans_dir"].glob("scan-*.csv")) == []


@pytest.mark.parametrize("argv", [["-p", "22"], ["--protocol", "udp"], ["-v"], ["--ai"],
                                  ["--abuseipdb"], ["--scan-unresponsive"],
                                  ["--export", "txt"]])
def test_scan_flags_need_a_target(scanner, fake, capsys, argv):
    with pytest.raises(SystemExit) as stop:
        scanner.main([*argv, AUTHORIZED])
    assert stop.value.code == 2
    assert "only with a TARGET" in capsys.readouterr().err
    assert fake["prompts"] == [] and fake["scans"] == []


@pytest.mark.parametrize("argv, words", [
    (["127.0.0.1", "-p", "0"], "between 1 and 65535"),
    (["10.0.0.0/33"], "not a valid network"),
    (["192.168.1.0/24", "--protocol", "both"], "TOOLKIT_SCAN_MAX_PROBES"),
    (["10.0.0.0/16"], "TOOLKIT_SCAN_MAX_HOSTS"),
])
def test_bad_input_and_limits_exit_2(scanner, fake, capsys, argv, words):
    assert scanner.main([*argv, AUTHORIZED]) == 2
    assert words in capsys.readouterr().out
    assert fake["scans"] == []


def test_wrong_choices_exit_2(scanner, fake):
    for argv in (["127.0.0.1", "--protocol", "icmp"], ["127.0.0.1", "--export", "pdf"]):
        with pytest.raises(SystemExit) as stop:
            scanner.main(argv)
        assert stop.value.code == 2


@pytest.mark.parametrize("stop", [EOFError, KeyboardInterrupt])
def test_cancel_at_the_question(scanner, fake, capsys, stop):
    fake["authorize"] = stop
    assert scanner.main(["127.0.0.1", "-p", "22"]) == 130
    out = capsys.readouterr().out
    assert "Scan cancelled." in out and "Traceback" not in out
    assert fake["scans"] == []


@pytest.mark.parametrize("stop", [EOFError, KeyboardInterrupt])
def test_cancel_in_the_interactive_questions(scanner, monkeypatch, capsys, stop):
    answers = iter(["127.0.0.1", "1", stop])

    def fake_input(prompt=""):
        answer = next(answers)
        if isinstance(answer, type):
            raise answer
        return answer

    monkeypatch.setattr(builtins, "input", fake_input)
    assert scanner.main([]) == 130
    assert "Scan cancelled." in capsys.readouterr().out


def test_ctrl_c_during_the_scan(fake_ports, scan_service, scanner, fake, monkeypatch, capsys):
    def interrupted(host, ports, protocols, progress=None):
        raise KeyboardInterrupt

    fake_ports(interrupted)
    assert scanner.main(["127.0.0.1", "-p", "22", AUTHORIZED]) == 130
    assert "Scan cancelled." in capsys.readouterr().out


def test_ctrl_c_stops_the_queued_checks(monkeypatch):
    """scan_ports: after Ctrl+C only the checks already running finish (not all 2,000)."""
    started = []
    lock = threading.Lock()

    def check(host, port):
        with lock:
            started.append(port)
        if port == 1:
            raise KeyboardInterrupt
        time.sleep(0.01)
        return PortResult(port, "tcp", "closed")

    monkeypatch.setattr(recon_scanner, "scan_tcp_port", check)
    with pytest.raises(KeyboardInterrupt):
        recon_scanner.scan_ports("127.0.0.1", range(1, 2001), ["tcp"])
    assert len(started) < 500


def test_cancelled_progress_does_not_say_100(scan_screen, scanner, capsys):
    progress = scan_screen.ConsoleProgress(interval=60)
    progress.start(4)
    progress.advance()
    progress.finish()
    out = capsys.readouterr().out
    assert out.endswith("\rProgress: 25.00% complete | 3 tasks remaining\n")
    assert "100.00%" not in out


def test_interactive_mode_unchanged(fake_ports, scan_service, scanner, monkeypatch, capsys):
    """No TARGET: the same questions in the same order as before step 1.4c."""
    prompts = []
    answers = {"Enter target": "127.0.0.1", "Enter start ": "20", "Enter end po": "22",
               "Select proto": "both", "\nOnly scan s": "yes", "Enable verbo": "n",
               "Do you want ": "y", "Enter desire": "txt"}

    def answer(prompt=""):
        prompts.append(prompt[:12])
        return answers[prompt[:12]]

    scans = []
    monkeypatch.setattr(builtins, "input", answer)
    monkeypatch.setattr(scan_service, "ping_host", lambda host, timeout=1: (True, 128))
    fake_ports(lambda host, ports, protocols, progress=None:
                        scans.append((list(ports), protocols)) or
                        [PortResult(21, "tcp", "open", "ftp", "")])
    monkeypatch.setattr(scan_service, "export_results", lambda entries, fmt: f"report.{fmt}")
    assert scanner.main([]) == 0
    assert prompts == ["Enter target", "Enter start ", "Enter end po", "Select proto",
                       "\nOnly scan s", "Enable verbo", "Do you want ", "Enter desire"]
    assert scans == [([20, 21, 22], ["tcp", "udp"])]
    assert "Results exported to report.txt" in capsys.readouterr().out


def test_every_scan_says_what_goes_online(scanner, fake, capsys):
    """Step A5: the plan before the first target, the requests at the end."""
    assert scanner.main(["127.0.0.1", "-p", "22", AUTHORIZED, "--ai", "--abuseipdb"]) == 0
    out = capsys.readouterr().out
    assert ("Online in this scan:\n"
            "  NVD (CVEs, services.nvd.nist.gov): on, the software name and version only; "
            "no NVD_API_KEY, so one request every 6 s\n"
            "  AI explanation (OpenRouter): off: no OPENROUTER_API_KEY\n"
            "  IP reputation (AbuseIPDB): off: no public address (a private one is never sent)\n"
            ) in out
    assert out.index("Online in this scan:") < out.index("Scanning target")
    assert "Requests sent online: NVD 1, AI (OpenRouter) 0, AbuseIPDB 0\n" in out
