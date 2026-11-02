"""Test helpers for the web scan (v5.2 step 2.4d): the network faked - the ping, the engine,
the DNS query, nmap's check and the online lookups record what they were asked; nothing leaves
this computer. ``fake_network(monkeypatch)`` puts them in place and returns the ``Recorder``."""

import threading

from toolkit.core.data.recon.findings import IntelFinding
from toolkit.core.data.recon.scanner import PortResult
from toolkit.core.services.recon import engines, targets
from toolkit.core.services.recon import scan as core_scan
from toolkit.core.services.recon.intel import Intel
from toolkit.web.services import scan

OPEN = [PortResult(22, "tcp", "open", "ssh", "SSH-2.0-OpenSSH_8.9p1"),
        PortResult(8080, "tcp", "open", "http", "\x1b[31mServer: nginx/1.25.3\x1b[0m")]


class Recorder:
    """What reached the network, by kind."""

    def __init__(self):
        self.sent = {"dns": [], "ping": [], "scan": [], "cves": [], "reputation": [],
                     "nmap_check": []}
        self.release = threading.Event()
        self.release.set()
        self.entered = threading.Event()
        self.down = set()

    def touched(self):
        return {kind: items for kind, items in self.sent.items() if items}


def _checks(rec, ports, progress, stop):
    """The fake engine's port checks: each counted, the stop looked at, then held until
    ``rec.release``."""
    rec.entered.set()
    progress.start(len(ports))
    for _ in ports:
        if stop is not None and stop.is_set():
            raise core_scan.ScanStopped()
        progress.advance()
    while not rec.release.wait(0.01):
        if stop is not None and stop.is_set():
            raise core_scan.ScanStopped()
    return list(OPEN)


class FakeEngine(engines.BuiltinEngine):
    rec = None

    def scan(self, host, ports, protocols, progress=None, stop=None):
        self.rec.sent["scan"].append(host)
        return _checks(self.rec, list(ports), progress, stop)


class FakeCves:
    name, kind = "Offline", "cves"

    def __init__(self, rec):
        self.rec = rec

    def lookup(self, subject):
        self.rec.sent["cves"].append(subject)
        return IntelFinding(cves=({"id": "CVE-2023-1", "cvss": 7.5, "cvss_version": "3.1",
                                   "severity": "HIGH", "match": "version"},), source="offline")


class FakeReputation(FakeCves):
    name, kind = "AbuseIPDB", "ip_reputation"

    def lookup(self, subject):
        self.rec.sent["reputation"].append(subject)
        return IntelFinding(data={"abuseConfidenceScore": 3, "totalReports": 1,
                                  "isp": "]0;ownedExample ISP"})


def fake_network(monkeypatch):
    """The fakes in place of the network; the Recorder of what they were asked."""
    rec = Recorder()
    rec.keys = {"openrouter": "", "abuseipdb": "", "nvd": ""}
    real_intel, real_problem = Intel, scan.engine_problem

    def ping(host):
        rec.sent["ping"].append(host)
        return host not in rec.down, 64

    def resolve(name):
        rec.sent["dns"].append(name)
        return "10.0.0.9"

    def engine_problem(*args):
        rec.sent["nmap_check"].append(args)
        return real_problem(*args)

    monkeypatch.setattr(FakeEngine, "rec", rec)
    monkeypatch.setitem(engines.ENGINES, "builtin", FakeEngine)
    monkeypatch.setattr(core_scan, "ping_host", ping)
    monkeypatch.setattr(core_scan, "Intel", lambda keys, use_ai=False, stop=None: real_intel(
        keys, use_ai, providers=[FakeCves(rec), FakeReputation(rec)], stop=stop))
    monkeypatch.setattr(targets, "first_address", resolve)
    monkeypatch.setattr(scan, "engine_problem", engine_problem)
    monkeypatch.setattr(scan, "load_api_keys", lambda: dict(rec.keys))
    return rec
