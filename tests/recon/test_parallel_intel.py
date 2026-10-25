"""Step 1.4e (R3): lookups run up to 4 at a time, each product once per scan (a failed
one is tried again), NVD's request spacing kept across threads. No network: fakes only."""

import threading
import time

import pytest
import requests

from toolkit.core.data.recon.findings import IntelFinding
from toolkit.core.services.recon import intel as recon_intel
from toolkit.core.data.recon import lookup, nvd
from toolkit.core.data.recon.scanner import PortResult


class SlowCves:
    """A CVE source that takes ``delay`` seconds and counts how many run at once."""

    name = "Slow"
    kind = "cves"

    def __init__(self, delay=0.1, fail=()):
        self.delay, self.fail = delay, set(fail)
        self.asked = []
        self.running = self.most = 0
        self.lock = threading.Lock()

    def lookup(self, subject):
        with self.lock:
            self.asked.append(subject)
            self.running += 1
            self.most = max(self.most, self.running)
        try:
            time.sleep(self.delay)
            if subject in self.fail:
                self.fail.discard(subject)  # fails once
                raise requests.Timeout("read timed out")
            return IntelFinding(cves=({"id": f"CVE-2026-{len(subject):04d}", "cvss": 5.0,
                                       "severity": "MEDIUM"},))
        finally:
            with self.lock:
                self.running -= 1


def test_at_most_four_at_a_time():
    assert lookup.INTEL_WORKERS == 4
    source = SlowCves()
    intel = recon_intel.Intel({}, providers=[source])
    products = [f"Product {n}.0" for n in range(8)]
    started = time.monotonic()
    found = intel.lookup_many(products)
    elapsed = time.monotonic() - started
    assert source.most == 4
    assert list(found) == products, "in the order given"
    assert all(found[p]["severity"] == "MEDIUM" for p in products)
    assert elapsed < 8 * 0.1 * 0.75, f"not parallel: {elapsed:.2f} s"


def test_each_product_once_per_scan():
    source = SlowCves(delay=0.05)
    intel = recon_intel.Intel({}, providers=[source])
    intel.lookup_many(["OpenSSH 8.9", "OpenSSH 8.9", "nginx 1.25"])
    intel.lookup_many(["OpenSSH 8.9", "Apache 2.4.49"])  # the next host
    assert sorted(source.asked) == ["Apache 2.4.49", "OpenSSH 8.9", "nginx 1.25"]


def test_a_failed_lookup_is_tried_again_on_the_next_host():
    source = SlowCves(delay=0, fail={"OpenSSH 8.9"})
    intel = recon_intel.Intel({}, providers=[source])
    first = intel.lookup_many(["OpenSSH 8.9"])["OpenSSH 8.9"]
    assert first["errors"] == ["Slow: timed out"] and first["severity"] == "UNKNOWN"
    second = intel.lookup_many(["OpenSSH 8.9"])["OpenSSH 8.9"]
    assert second["errors"] == [] and second["severity"] == "MEDIUM"
    assert source.asked == ["OpenSSH 8.9", "OpenSSH 8.9"]


def test_results_in_the_order_given():
    class SlowFirst(SlowCves):
        def lookup(self, subject):
            self.delay = 0.15 if subject == "Alpha 1.0" else 0.0
            return super().lookup(subject)

    found = recon_intel.Intel({}, providers=[SlowFirst()]).lookup_many(["Alpha 1.0", "Beta 1.0"])
    assert list(found) == ["Alpha 1.0", "Beta 1.0"]


def test_nothing_to_look_up():
    assert recon_intel.Intel({}, providers=[SlowCves()]).lookup_many([]) == {}


def test_nvd_spacing_across_threads(monkeypatch, fake_session):
    """Each request starts at least NVD_INTERVAL after the previous request's start, however
    many threads ask (so never more than NVD's limit per 30 s)."""
    monkeypatch.setattr(nvd, "NVD_INTERVAL", {False: 0.2, True: 0.02})
    starts = []
    session = fake_session(get={"vulnerabilities": []})
    answer = session.get

    def recording_get(*args, **kwargs):
        starts.append(time.monotonic())
        return answer(*args, **kwargs)

    session.get = recording_get
    intel = recon_intel.Intel({}, session=session)
    intel.lookup_many([f"Product {n}.0" for n in range(5)])
    starts.sort()
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    assert len(starts) == 5
    assert min(gaps) >= 0.2 - 0.02, gaps


def test_nvd_cache_is_shared_by_threads(fake_session, no_nvd_wait):
    session = fake_session(get={"vulnerabilities": []})
    provider = nvd.NvdProvider(session)
    threads = [threading.Thread(target=provider.lookup, args=("nginx 1.25",)) for _ in range(4)]
    provider.lookup("nginx 1.25")
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    asked = [call[2]["params"].get("virtualMatchString") or call[2]["params"]["keywordSearch"]
             for call in session.calls]
    assert asked == ["cpe:2.3:a:f5:nginx:1.25", "nginx 1.25"], \
        "one lookup (the version match, then the keyword search), the others from the cache"


def test_ctrl_c_cancels_the_queued_lookups():
    asked = []

    class Interrupted:
        name, kind = "Stop", "cves"

        def lookup(self, subject):
            asked.append(subject)
            if subject == "Product 0.0":
                raise KeyboardInterrupt
            time.sleep(0.05)
            return IntelFinding(cves=())

    intel = recon_intel.Intel({}, providers=[Interrupted()])
    with pytest.raises(KeyboardInterrupt):
        intel.lookup_many([f"Product {n}.0" for n in range(100)])
    assert len(asked) < 50


def test_scan_prints_entries_in_port_order(fake_ports, scan_service, scanner, monkeypatch, capsys):
    """Lookups finish in any order; the report keeps the port order, and every port of a
    shared product gets its result."""
    delays = {"OpenSSH 8.9": 0.15, "Apache 2.4.49": 0.0, "vsFTPd 3.0.3": 0.05}

    class Uneven:
        name, kind = "Uneven", "cves"
        asked, running, most = [], [0], [0]
        lock = threading.Lock()

        def lookup(self, subject):
            with self.lock:
                self.asked.append(subject)
                self.running[0] += 1
                self.most[0] = max(self.most[0], self.running[0])
            time.sleep(delays[subject] + 0.05)
            with self.lock:
                self.running[0] -= 1
            return IntelFinding(cves=({"id": "CVE-2026-0001", "cvss": 9.8,
                                       "severity": "CRITICAL"},))

    source = Uneven()
    real_intel = recon_intel.Intel
    monkeypatch.setattr(scan_service, "Intel", lambda keys, use_ai=False, stop=None:
                        real_intel(keys, use_ai, providers=[source], stop=stop))
    open_ports = [PortResult(21, "tcp", "open", "ftp", "220 (vsFTPd 3.0.3)"),
                  PortResult(22, "tcp", "open", "ssh", "SSH-2.0-OpenSSH_8.9p1"),
                  PortResult(80, "tcp", "open", "http", "Apache/2.4.49 (Unix)"),
                  PortResult(2222, "tcp", "open", "", "SSH-2.0-OpenSSH_8.9p1"),
                  PortResult(9000, "tcp", "open", "", "hello")]
    fake_ports(lambda *args, **kwargs: open_ports)
    monkeypatch.setattr(scan_service, "ping_host", lambda host, timeout=1: (True, 64))
    assert scanner.main(["127.0.0.1", "-p", "1-9000", "--i-am-authorized"]) == 0
    out = capsys.readouterr().out
    assert "Looking up 3 products online (up to 4 at a time)..." in out
    ports = [line.split()[1] for line in out.splitlines() if line.startswith("Port: ")]
    assert ports == ["21/tcp", "22/tcp", "80/tcp", "2222/tcp", "9000/tcp"]
    assert sorted(source.asked) == ["Apache 2.4.49", "OpenSSH 8.9", "vsFTPd 3.0.3"]
    assert source.most[0] == 3, "the scan looks its products up at the same time"
    assert out.count("CVE-2026-0001 (CVSS 9.8, CRITICAL)") == 4  # not the unknown banner
