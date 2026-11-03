"""v5.2 step 2.4d: the web scan's service - the scan contract (nothing sent before the
authorization, not even DNS), the checks, the job (one scan per session, cancel, the time
limit), what the page sees while it runs, the answer and the report download. The network is
faked (``web_scan.fake_network``): nothing leaves this computer.
"""

import logging
import os
import time

import pytest

from toolkit.core.data import logs
from toolkit.core.data.settings import ScanConfig
from toolkit.core.services.recon import targets
from toolkit.core.services.recon import scan as core_scan
from toolkit.web.services import jobs, scan, scan_view
from toolkit.web.services.errors import ServiceError
from toolkit.web.services.workspace import Workspace
from web_scan import fake_network

LIMITS = ScanConfig()


@pytest.fixture
def net(monkeypatch):
    return fake_network(monkeypatch)


@pytest.fixture
def runner():
    return jobs.JobRunner(jobs.SCAN)


def _choices(target="10.0.0.5", **options):
    return scan.ScanChoices(target=target, ports=options.pop("ports", "22,8080"),
                            authorized=options.pop("authorized", True), **options)


def _wait(job, timeout=30):
    deadline = time.monotonic() + timeout
    while job.state not in jobs.FINISHED:
        assert time.monotonic() < deadline, "the scan job didn't end"
        time.sleep(0.01)
    return job


def _refused(ws, choices, runner, limits=LIMITS):
    with pytest.raises(ServiceError) as refused:
        scan.start_scan(ws, choices, runner, limits)
    return refused.value


# --- Nothing before the authorization ------------------------------------------------------

@pytest.mark.parametrize("target", ["10.0.0.5", "scanme.example", "10.0.0.0/24", "::1"])
def test_nothing_is_sent_before_the_authorization(net, runner, target):
    ws = Workspace()
    refused = _refused(ws, _choices(target, authorized=False, engine="nmap"), runner)
    assert refused.code == "NOT_AUTHORIZED" and "authorised" in refused.message
    assert net.touched() == {}, "not even DNS, not even nmap's own --version"
    assert ws.jobs == {} and not ws.scanning.locked()


def test_the_authorization_is_checked_first(net, runner):
    refused = _refused(Workspace(), _choices(protocol="smtp", ports="0", engine="x",
                                             authorized=False), runner)
    assert refused.code == "NOT_AUTHORIZED"


@pytest.mark.parametrize("options, words", [
    ({"protocol": "icmp"}, "'protocol' must be one of tcp, udp, both"),
    ({"ports": "70000"}, "between 1 and 65535"),
    ({"ports": "22,,80"}, "empty item"),
    ({"engine": "masscan"}, "'engine' must be one of builtin, nmap"),
    ({"timing": "aggressive"}, "choose the nmap engine"),
    ({"scripts": True}, "choose the nmap engine"),
    ({"engine": "nmap", "service_detection": "max"}, "'service_detection' must be one of"),
    ({"engine": "nmap", "timing": "insane"}, "'timing' must be one of"),
])
def test_bad_choices_are_refused_before_any_network(net, runner, options, words):
    refused = _refused(Workspace(), _choices("scanme.example", **options), runner)
    assert refused.code == "BAD_REQUEST" and words in refused.message
    assert net.touched() == {}


def test_a_bad_target_is_refused_after_the_dns_query(net, runner, monkeypatch):
    def unknown(name):
        net.sent["dns"].append(name)
        raise OSError("not found")

    monkeypatch.setattr(targets, "first_address", unknown)
    refused = _refused(Workspace(), _choices("no-such-host.example"), runner)
    assert refused.code == "BAD_REQUEST" and "Could not resolve" in refused.message
    assert set(net.touched()) == {"nmap_check", "dns"}, "the engine check, then the DNS query"
    assert net.sent["dns"] == ["no-such-host.example"]


@pytest.mark.parametrize("target, ports, words", [
    ("10.0.0.0/16", "22", "TOOLKIT_SCAN_MAX_HOSTS"),
    ("10.0.0.0/30", "1-65535", "TOOLKIT_SCAN_MAX_PROBES"),
])
def test_a_scan_above_the_limits_is_refused(net, runner, target, ports, words):
    ws = Workspace()
    refused = _refused(ws, _choices(target, ports=ports), runner)
    assert refused.code == "SCAN_LIMIT" and words in refused.message
    assert net.sent["ping"] == net.sent["scan"] == [] and not ws.scanning.locked()


def test_nmap_missing_is_refused_without_this_computers_paths(net, runner):
    ws = Workspace()
    refused = _refused(ws, _choices(engine="nmap"), runner)
    assert refused.code == "ENGINE_UNAVAILABLE"
    assert "nmap can't be used on this computer" in refused.message
    assert os.environ["TOOLKIT_NMAP_PATH"] not in refused.message, "not where it looked"
    assert net.sent["ping"] == [] and not ws.scanning.locked()


# --- The scan as a job ---------------------------------------------------------------------

def test_a_scan_runs_and_its_answer_is_kept(net, runner, caplog):
    ws = Workspace()
    caplog.set_level(logging.INFO, "toolkit.core.services.recon.record")
    job = scan.start_scan(ws, _choices("scanme.example"), runner, LIMITS)
    assert ws.scanning.locked() and ws.leased
    _wait(job)
    assert job.state == "done", job.error
    assert {kind: items for kind, items in net.touched().items() if kind != "nmap_check"} == {
        "dns": ["scanme.example"], "ping": ["10.0.0.9"], "scan": ["10.0.0.9"],
        "cves": ["OpenSSH 8.9", "nginx 1.25.3"]}
    answer = job.result
    assert answer["message"] == "Scanned 1 host(s): 2 open port(s)."
    assert [(e["port"], e["software"], e["severity"]) for e in answer["entries"]] == [
        (22, "OpenSSH 8.9", "HIGH"), (8080, "nginx 1.25.3", "HIGH")]
    assert "\x1b" not in str(answer), "network text is cleaned"
    assert answer["hosts"] == [{"target": "10.0.0.9", "state": "scanned",
                                "os": answer["hosts"][0]["os"]}]
    assert answer["summary"]["open_ports"] == 2 and answer["engine"] == "builtin"
    assert answer["online"]["abuseipdb"].startswith("off")
    assert (job.done, job.total) == (1000, 1000)
    assert ws.scan.answer is answer and ws.scan.entries == answer["entries"]
    assert not ws.scanning.locked() and not ws.leased
    lines = job.notes()
    assert lines[0].startswith("scanme.example resolved to 10.0.0.9")
    assert "Scanning 10.0.0.9" in lines and any(line.startswith("Open: 22/tcp ssh")
                                                for line in lines)
    assert lines[-1].startswith("Done: 1 host(s), 2 open port(s)")
    assert job.notes(len(lines) - 1) == lines[-1:]
    # the record of the scan (HS3, tests/recon/test_scan_record.py), not a line of the job's
    logged = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Scan ")]
    assert [line.split(" ", 2)[1] for line in logged] == ["started", "ended"]
    assert '"target": "scanme.example", "addresses": ["10.0.0.9"]' in logged[0]
    assert '"outcome": "done", "open_ports": 2' in logged[1]


def test_hosts_that_dont_answer_are_skipped_unless_asked(net, runner):
    net.down = {"10.0.0.1"}
    job = _wait(scan.start_scan(Workspace(), _choices("10.0.0.0/30"), runner, LIMITS))
    assert [(h["target"], h["state"]) for h in job.result["hosts"]] == [
        ("10.0.0.1", "skipped"), ("10.0.0.2", "scanned")]
    assert net.sent["scan"] == ["10.0.0.2"]
    again = _wait(scan.start_scan(Workspace(), _choices("10.0.0.0/30", scan_unresponsive=True),
                                  runner, LIMITS))
    assert [h["state"] for h in again.result["hosts"]] == ["scanned", "scanned"]


def test_the_phase_and_progress_go_host_by_host(net, runner):
    net.release.clear()
    job = scan.start_scan(Workspace(), _choices("10.0.0.0/30"), runner, LIMITS)
    assert net.entered.wait(30)
    deadline = time.monotonic() + 10
    while job.done < 1000 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert (job.done, job.total) == (1000, 2000), "all of the first host's ports, half the scan"
    assert job.phase() == "Host 1 of 2: checking the ports of 10.0.0.1"
    net.release.set()
    _wait(job)
    assert (job.done, job.total) == (2000, 2000)


def test_one_scan_at_a_time_in_a_session(net, runner):
    ws = Workspace()
    net.release.clear()
    first = scan.start_scan(ws, _choices(), runner, LIMITS)
    refused = _refused(ws, _choices(), runner)
    assert refused.code == "BUSY"
    other = scan.start_scan(Workspace(), _choices("10.0.0.6"), runner, LIMITS)
    assert other.state == "queued", "another session's scan waits in the queue"
    net.release.set()
    assert _wait(first).state == _wait(other).state == "done"
    assert _wait(scan.start_scan(ws, _choices(), runner, LIMITS)).state == "done"


def test_a_cancelled_scan_keeps_nothing(net, runner):
    ws = Workspace()
    previous = _wait(scan.start_scan(ws, _choices(), runner, LIMITS)).result
    net.release.clear()
    net.entered.clear()
    job = scan.start_scan(ws, _choices("10.0.0.6"), runner, LIMITS)
    assert net.entered.wait(30)
    runner.cancel(job)
    _wait(job)
    assert (job.state, job.error) == ("cancelled", {
        "code": "CANCELLED", "message": "Scan cancelled. Nothing was kept from it."})
    assert ws.scan.answer is previous, "the last finished scan stays"
    assert not ws.scanning.locked() and not ws.leased


def test_a_scan_past_its_time_limit(net, runner):
    net.release.clear()
    job = _wait(scan.start_scan(Workspace(), _choices(), runner, ScanConfig(max_minutes=0)))
    assert job.error == {"code": "TIME_LIMIT", "message": "Scan stopped after 0 minutes "
                                                          "(TOOLKIT_SCAN_MAX_MIN). Nothing was kept."}


def test_a_private_address_never_goes_to_abuseipdb(net, runner):
    net.keys["abuseipdb"] = "ab-test-key-123456"
    job = _wait(scan.start_scan(Workspace(), _choices("10.0.0.5", abuseipdb=True), runner,
                                LIMITS))
    assert job.state == "done" and net.sent["reputation"] == []
    public = _wait(scan.start_scan(Workspace(), _choices("8.8.8.8", abuseipdb=True), runner,
                                   LIMITS))
    assert net.sent["reputation"] == ["8.8.8.8"]
    assert public.result["reputation"]["8.8.8.8"]["isp"] == "Example ISP"
    assert "ab-test-key-123456" not in str(public.result) + str(public.notes())
    assert logs.redact("key ab-test-key-123456") == "key [REDACTED]", "kept out of the log"


# --- The report download and the options ---------------------------------------------------

def test_the_report_of_the_last_scan(net, runner):
    ws = Workspace()
    with pytest.raises(ServiceError) as none_yet:
        scan.scan_report(ws, "txt")
    assert none_yet.value.code == "NO_SCAN"
    _wait(scan.start_scan(ws, _choices(), runner, LIMITS))
    for fmt, media in (("txt", "text/plain"), ("json", "application/json"),
                       ("csv", "text/csv")):
        text, media_type, name = scan.scan_report(ws, fmt)
        assert text == core_scan.report_text(ws.scan.entries, fmt) and media_type == media
        assert name == f"scan-{ws.scan.stamp}.{fmt}" and "OpenSSH 8.9" in text
    with pytest.raises(ServiceError, match="'format' must be one of txt, json, csv"):
        scan.scan_report(ws, "exe")


def test_the_options_say_whether_each_key_is_set_never_the_key(net):
    net.keys.update(openrouter="or-secret-value-1", nvd="")
    options = scan.scan_options(net.keys, LIMITS)
    assert options["keys"] == {"openrouter": True, "abuseipdb": False, "nvd": False}
    assert "or-secret-value-1" not in str(options)
    builtin, nmap = options["engines"]
    assert builtin == {"id": "builtin", "available": True, "reason": None}
    assert nmap["available"] is False and os.environ["TOOLKIT_NMAP_PATH"] not in nmap["reason"]
    assert options["limits"] == {"max_hosts": 256, "max_checks": 65536, "max_minutes": 60}
    assert options["default_ports"] == "1-1024" and options["protocols"] == ["tcp", "udp",
                                                                              "both"]


def test_engine_lines_never_show_this_computers_paths():
    assert scan_view._without_paths(r"nmap 7.92 (C:\Program Files (x86)\Nmap\nmap.exe)") == \
        "nmap 7.92"
    assert scan_view._without_paths("nmap: nmap was not found (looked in C:\\x).").startswith(
        "nmap: nmap can't be used on this computer.")
    assert scan_view._without_paths("raw packets: yes (SYN scan)") == "raw packets: yes (SYN scan)"


# --- The job's log -------------------------------------------------------------------------

def test_a_jobs_log_stops_at_its_limit(monkeypatch):
    monkeypatch.setattr(jobs, "MAX_LOG_LINES", 3)
    job = jobs.Job(kind=jobs.SCAN, work=lambda job: {}, on_end=lambda: None, max_seconds=60,
                   steps=lambda done, total: "")
    for n in range(6):
        job.note(f"line {n}")
    assert job.notes() == ["line 0", "line 1", "line 2",
                           "(the log stops at 3 lines; the result has everything)"]
    assert job.notes(2) == job.notes()[2:]
    job.tally(7, 5)
    assert (job.done, job.total) == (5, 5)
