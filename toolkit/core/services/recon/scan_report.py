"""A saved scan's report (v5.4 F5): what the HTML report shows, as plain data.

1. the scan: when, from where, the target, the ports, the engine, what was looked up online;
2. the summary: the hosts, the open ports, the ports by severity, the ports to look at first;
3. each host, and each open port with its CVEs, how each was found, and its advice;
4. what the report can't say.

Every decision is taken here (the words, the advice, which link is made); the page only shows
it. See also: data/recon/report_render.py (the HTML), data/recon/scan_report.html (the page's
order), data/recon/scan_report.css (its look), DECISIONS F5.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...data.recon import report_render
from ...data.recon.findings import MATCH_LABELS
from ...data.recon.text import clean_text, one_line
from .saved_scans import HOST_STATES, ONLINE, WORST_FIRST, saved_scan, scan_view, summary, when

NVD_PAGE = "https://nvd.nist.gov/vuln/detail/"
CVE_ID = re.compile(r"CVE-\d{4}-\d{4,7}")  # the only text a link is made from
FIRST = ("CRITICAL", "HIGH")  # the ports listed under "Look at these first"
LEVELS = {"CRITICAL": "critical", "HIGH": "high", "MEDIUM": "medium", "LOW": "low",
          "NOT CHECKED": "unchecked"}  # severity -> the page's CSS class; anything else "none"
NAMED_CVES = 3  # CVEs named in one advice
SHOWN_CVES = 50  # CVEs listed for one port; the JSON export has them all


def _cve(found: Mapping[str, Any]) -> dict[str, Any]:
    """One CVE: its id, score, severity, how it was found, and its NVD page (a real id only)."""
    cve_id = one_line(found["id"], 40)
    return {"id": cve_id, "cvss": found["cvss"], "cvss_version": one_line(found["cvss_version"], 8),
            "severity": one_line(found["severity"], 20),
            "how": MATCH_LABELS.get(found["match"], "match unknown"),
            "confirmed": found["match"] == "version",
            "url": NVD_PAGE + cve_id if CVE_ID.fullmatch(cve_id) else None}


def _ids(cves: list[dict[str, Any]]) -> str:
    """The worst CVEs' ids, then how many more."""
    worst = sorted(cves, key=lambda c: -(c["cvss"] or 0))
    named = ", ".join(c["id"] for c in worst[:NAMED_CVES])
    return named + (f" and {len(cves) - NAMED_CVES} more" if len(cves) > NAMED_CVES else "")


def _advice(port: Mapping[str, Any], day: str) -> list[str]:
    """What to do about one port, tied to its CVEs; never "clean" when nothing was checked."""
    where, software = f"{port['port']}/{port['protocol']}", port["software"]
    if not software:
        return [f"The software on this port wasn't recognised, so no CVE was looked up. Check "
                f"what runs on {where} and whether it needs to be open."]
    if port["severity"] == "NOT CHECKED":
        return [f"The CVEs of {software} were not looked up in this scan (NVD off, or no answer): "
                "this is not a clean result. Look them up before relying on it."]
    confirmed = [c for c in port["cves"] if c["confirmed"]]
    by_name = [c for c in port["cves"] if not c["confirmed"]]
    advice = []
    if confirmed:
        advice.append(f"Update {software} to a version that fixes {_ids(confirmed)} (each CVE's "
                      f"NVD page lists the fixed versions), or limit who can reach {where} until "
                      "then.")
    if by_name:
        advice.append(f"Check the exact version of {software} first: {_ids(by_name)} "
                      f"{'was' if len(by_name) == 1 else 'were'} found by its name only and may "
                      "not apply to it.")
    if not port["cves"]:
        advice.append(f"NVD listed no CVE for {software} on {day}. Keep it updated, and close "
                      "the port if it isn't needed.")
    return advice


def _port(entry: Mapping[str, Any], anchor: str, day: str) -> dict[str, Any]:
    """One open port: its service, software, CVEs, the AI's text, failed lookups and advice."""
    cves = [_cve(c) for c in entry["cves"]]
    severity = one_line(entry["severity"], 20)
    port = {
        "anchor": anchor, "target": one_line(entry["target"], 100), "port": entry["port"],
        "protocol": one_line(entry["protocol"], 5), "service": one_line(entry["service"], 60),
        "banner": one_line(entry["banner"], 300), "software": one_line(entry["software"], 100),
        "cpe": [one_line(name, 120) for name in entry["cpe"]],
        "severity": severity if entry["software"] else "no lookup",
        "unconfirmed": bool(entry["severity_unconfirmed"]),
        "level": LEVELS.get(severity, "none") if entry["software"] else "none",
        "max_cvss": entry["max_cvss"], "cve_source": one_line(entry["cve_source"], 200),
        "cves": cves[:SHOWN_CVES], "more_cves": max(len(cves) - SHOWN_CVES, 0),
        "ai": clean_text(entry["ai_explanation"], 4000).strip(),
        "errors": [one_line(error, 300) for error in entry["lookup_errors"]],
    }
    port["advice"] = _advice({**port, "cves": cves, "severity": severity}, day)
    return port


def _reputation(found: Mapping[str, Any] | None) -> str:
    """AbuseIPDB's answer for a host, in a line ("" when it wasn't asked)."""
    if not found:
        return ""
    if "error" in found:
        return "AbuseIPDB: the lookup failed (" + one_line(found["error"], 200) + ")"
    return (f"AbuseIPDB: confidence of abuse {one_line(found.get('abuseConfidenceScore'), 10)}%, "
            f"{one_line(found.get('totalReports'), 10)} report(s)")


def _host(host: Mapping[str, Any], reputation: Mapping[str, Any]) -> dict[str, Any]:
    """One host: how it ended (and why it failed), its OS hint and AbuseIPDB's answer."""
    why = one_line(host.get("message") or "", 300) if host["state"] == "failed" else ""
    return {"address": one_line(host["target"], 100),
            "state": HOST_STATES.get(host["state"], one_line(host["state"], 20)), "why": why,
            "os": one_line(host["os"], 300),
            "reputation": _reputation(reputation.get(host["target"]))}


def _by_severity(ports: list[dict[str, Any]]) -> str:
    """``3 open; CRITICAL 1, HIGH 1, LOW 1``, the worst first."""
    labels = [p["severity"] for p in ports]
    counts = [f"{level} {labels.count(level)}" for level in (*WORST_FIRST, "no lookup")
              if level in labels]
    return f"{len(ports)} open" + ("; " + ", ".join(counts) if counts else "")


def _limits(record: Mapping[str, Any], head: Mapping[str, Any], day: str) -> list[str]:
    """What the report can't say."""
    skipped = head["hosts"]["skipped"]
    limits = [
        "A CVE found by the software's version is a possible match to check, not a confirmed "
        "vulnerability: a fix the vendor backported, or a setting, can mean it doesn't apply.",
        "A CVE found by the software's name only (unconfirmed) was not checked against its "
        "version.",
        f"Only ports {one_line(record['ports'], 200)} "
        f"({' + '.join(record['protocols']).upper()}) were checked; a closed or filtered port "
        "isn't listed.",
        "The AI's explanations may be wrong.",
        f"It shows the network as it was on {day}.",
    ]
    if skipped:
        limits.insert(3, f"{skipped} host{'' if skipped == 1 else 's'} didn't answer ping and "
                         f"{'was' if skipped == 1 else 'were'} skipped: "
                         f"{'it was' if skipped == 1 else 'they were'} not scanned.")
    return limits


def report_model(record: Mapping[str, Any]) -> dict[str, Any]:
    """The report of a scan record (``saved_scans.saved_scan`` checks one read from disk):
    the scan, the summary, the ports to look at first, the hosts, the ports, the limits."""
    head, view = summary(record), scan_view(record)
    day = record["started"][:10]
    ports = [_port(entry, f"port-{i}", day) for i, entry in enumerate(record["entries"], 1)]
    totals = record["summary"]
    return {
        "scan_id": record["scan_id"], "started": when(record["started"]),
        "seconds": record["seconds"], "source": one_line(record["source"], 60),
        "target": one_line(record["target"], 200),
        "addresses": ", ".join(one_line(a, 100) for a in record["addresses"]),
        "ports": one_line(record["ports"], 200), "port_count": head["port_count"],
        "protocols": " + ".join(record["protocols"]).upper(),
        "engine": one_line(view["engine_notice"][0], 200),
        "online": [(name, one_line(record["online"].get(key, "?"), 200)) for key, name in ONLINE],
        "sent": ", ".join(f"{one_line(name, 40)} {count}" for name, count in totals["sent"].items()),
        "failed_lookups": [one_line(text, 300) for text in totals["failed_lookups"]],
        "unavailable": [f"{one_line(name, 40)} ({one_line(why, 200)})"
                        for name, why in totals["unavailable"].items()],
        "hosts": [_host(host, record["reputation"]) for host in record["hosts"]],
        "host_counts": ", ".join(f"{n} {state}" for state, n in head["hosts"].items() if n),
        "by_severity": _by_severity(ports),
        "first": sorted((p for p in ports if p["severity"] in FIRST),
                        key=lambda p: (FIRST.index(p["severity"]), -(p["max_cvss"] or 0))),
        "ports_found": ports, "limits": _limits(record, head, day),
    }


def saved_report_html(scan_id: str, directory: str | Path | None = None,
                      stylesheet_url: str | None = None) -> str:
    """The report page of a saved scan (``report_render.report_html``: one file, or with
    ``stylesheet_url`` the web app's page); the errors of ``saved_scans.saved_scan``."""
    return report_render.report_html(report_model(saved_scan(scan_id, directory)),
                                     stylesheet_url)


def report_stylesheet() -> str:
    """The report's CSS, for the web app's page."""
    return report_render.stylesheet()
