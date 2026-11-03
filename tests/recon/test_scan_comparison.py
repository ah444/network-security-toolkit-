"""F6c: two saved scans compared, address by address and port by port. A port open in one scan
only is opened or closed only when the other scan checked it, else "not scanned" there; CVEs
are compared only when both scans looked the port's software up (NVD off in one: "not
checked", never "gone"); a host that one scan skipped or couldn't scan is not compared; scans
of different targets are compared on the addresses both scanned, and say so."""

import pytest

from toolkit.core.services.recon import saved_scans, scan_comparison

from .test_saved_scans import A, B, entry, record, save


def cve(cve_id, cvss=7.5):
    return {"id": cve_id, "cvss": cvss, "severity": "HIGH", "match": "version"}


def port(number, **over):
    made = entry(number)
    made.update(over)
    return made


def compare(a_over, b_over):
    return scan_comparison.compare_scans(record(A, **a_over), record(B, **b_over))


def one_host(**over):
    return {"target": "10.0.0.5", "addresses": ["10.0.0.5"],
            "hosts": [{"target": "10.0.0.5", "state": "scanned", "os": "", "message": ""}], **over}


def rows(answer, address="10.0.0.5"):
    [host] = [h for h in answer["hosts"] if h["address"] == address]
    return {(r["port"], r["protocol"]): r for r in host["rows"]}


def test_opened_closed_same_and_changed():
    answer = compare(one_host(ports="22,80,443", entries=[port(22), port(80)]),
                     one_host(ports="22,80,443", entries=[
                         port(22), port(443, service="https"),
                         port(80, software="nginx 1.25.3", service="http")]))
    found = rows(answer)
    assert {key: r["status"] for key, r in found.items()} == {
        (22, "tcp"): "same", (80, "tcp"): "changed", (443, "tcp"): "opened"}
    assert found[(80, "tcp")]["changes"] == ["service", "software"]
    assert found[(443, "tcp")]["a"] is None and found[(443, "tcp")]["b"]["service"] == "https"
    closed = rows(compare(one_host(ports="22,80", entries=[port(22), port(80)]),
                          one_host(ports="22,80", entries=[port(22)])))
    assert closed[(80, "tcp")]["status"] == "closed" and closed[(80, "tcp")]["b"] is None


def test_a_port_the_other_scan_didnt_check_is_not_scanned_there_not_closed():
    found = rows(compare(one_host(ports="22,8080", entries=[port(22), port(8080)]),
                         one_host(ports="22,443", entries=[port(22), port(443)])))
    assert found[(8080, "tcp")]["status"] == "not_scanned_in_b"
    assert found[(443, "tcp")]["status"] == "not_scanned_in_a"
    udp = rows(compare(one_host(ports="53", protocols=["tcp"], entries=[]),
                       one_host(ports="53", protocols=["tcp", "udp"],
                                entries=[port(53, protocol="udp")])))
    assert udp[(53, "udp")]["status"] == "not_scanned_in_a"


def test_cves_added_and_gone_when_both_scans_looked_them_up():
    found = rows(compare(
        one_host(entries=[port(22, cves=[cve("CVE-1"), cve("CVE-2", 9.8)], max_cvss=9.8,
                               severity="CRITICAL")]),
        one_host(entries=[port(22, cves=[cve("CVE-2", 9.8), cve("CVE-3")], max_cvss=9.8,
                               severity="CRITICAL")])))
    row = found[(22, "tcp")]
    assert (row["status"], row["cves_added"], row["cves_gone"], row["cves_note"]) == (
        "changed", ["CVE-3"], ["CVE-1"], "")
    assert row["changes"] == []  # the same service, software, severity and highest CVSS


def test_cves_not_checked_in_one_scan_are_never_gone():
    found = rows(compare(
        one_host(entries=[port(22, cves=[cve("CVE-1")], max_cvss=7.5, severity="HIGH")]),
        one_host(entries=[port(22, severity="NOT CHECKED")])))
    row = found[(22, "tcp")]
    assert row["cves_added"] is None and row["cves_gone"] is None
    assert row["cves_note"] == "not checked in B" and row["status"] == "changed"
    assert row["changes"] == ["severity", "max_cvss"]
    unknown = rows(compare(one_host(entries=[port(22, software="")]),
                           one_host(entries=[port(22, software="")])))
    assert unknown[(22, "tcp")]["cves_note"] == "software not recognised"
    assert unknown[(22, "tcp")]["status"] == "same"


def test_a_host_one_scan_didnt_scan_is_not_compared():
    skipped = [{"target": "10.0.0.5", "state": "skipped", "os": "", "message": ""}]
    answer = compare(one_host(entries=[port(22)]), one_host(hosts=skipped, entries=[]))
    [host] = answer["hosts"]
    assert (host["a"], host["b"], host["compared"], host["rows"]) == (
        "scanned", "skipped", False, [])
    assert answer["totals"]["closed"] == 0


def test_different_targets_compare_the_addresses_both_scanned():
    answer = compare(
        {"target": "10.0.0.4/31", "addresses": ["10.0.0.4", "10.0.0.5"]},
        {"target": "10.0.0.5", "addresses": ["10.0.0.5"],
         "hosts": [{"target": "10.0.0.5", "state": "scanned", "os": "", "message": ""}]})
    assert answer["same_target"] is False
    assert (answer["common"], answer["only_in_a"], answer["only_in_b"]) == (
        ["10.0.0.5"], ["10.0.0.4"], [])
    assert [h["address"] for h in answer["hosts"]] == ["10.0.0.5"]
    apart = compare({"target": "10.0.0.9", "addresses": ["10.0.0.9"],
                     "hosts": [{"target": "10.0.0.9", "state": "scanned", "os": "",
                                "message": ""}], "entries": []}, one_host())
    assert apart["common"] == [] and apart["hosts"] == []


def test_the_totals_and_both_summaries():
    answer = compare(one_host(ports="22,80,443", entries=[port(22), port(80)]),
                     one_host(ports="22,80,443", entries=[port(22, service="x"), port(443)]))
    assert answer["totals"] == {"opened": 1, "closed": 1, "changed": 1, "same": 0,
                                "not_scanned": 0, "cves_added": 0, "cves_gone": 0}
    assert (answer["a"]["scan_id"], answer["b"]["scan_id"]) == (A, B)


def test_two_saved_scans_by_id():
    save(A, **one_host(entries=[port(22)]))
    save(B, **one_host(entries=[]))
    answer = scan_comparison.compare_saved(A, B)
    assert rows(answer)[(22, "tcp")]["status"] == "closed"
    with pytest.raises(saved_scans.NotAScan):
        scan_comparison.compare_saved(A, "../x")
    with pytest.raises(saved_scans.NotAScan):
        scan_comparison.compare_saved(A, A)
    with pytest.raises(FileNotFoundError):
        scan_comparison.compare_saved(A, "20261109T100000Z-0000000f")
