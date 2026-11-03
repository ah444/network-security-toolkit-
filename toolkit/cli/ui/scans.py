"""What ``toolkit scan --history``, ``--show``, ``--compare`` and ``--delete`` print (v5.4 F6d):
the scan history's records, in the words of the web page's Saved Scans (scans.js,
scancompare.js). Text from the network is printed on one line, cleaned (``one_line``), as a
scan prints it; ``--delete`` names the record and asks before anything is removed."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ...core.services.recon.saved_scans import HOST_STATES, ONLINE, when
from ...core.services.recon.scan import one_line
from .scan import print_entry

NO_SCANS = ("No saved scans yet: each scan that finishes, here or on the web page, is saved in "
            "instance/scan_history.")
ENGINES = {"builtin": "built-in", "nmap": "nmap"}
STATUS = {"opened": "opened", "closed": "closed", "changed": "changed", "same": "the same",
          "not_scanned_in_a": "open in B; A didn't check this port",
          "not_scanned_in_b": "open in A; B didn't check this port"}
CHANGE_NAMES = {"service": "service", "software": "software", "severity": "severity",
                "max_cvss": "highest CVSS"}
YES = ("y", "yes")


def protocols(names: Sequence[str]) -> str:
    """``TCP`` or ``TCP + UDP``."""
    return " + ".join(names).upper()


def worst(scan: Mapping[str, Any]) -> str:
    """The worst severity of the ports looked up, as the page says it."""
    if scan["worst"] is None:
        return "no lookup" if scan["open_ports"] else "–"
    return str(scan["worst"]) + (" (unconfirmed)" if scan["worst_unconfirmed"] else "")


def scan_line(scan: Mapping[str, Any]) -> str:
    """One saved scan of the list."""
    hosts = ", ".join(f"{count} {state}" for state, count in scan["hosts"].items() if count)
    return (f"{scan['scan_id']}  {when(scan['started'])}  {one_line(scan['target'], 100)}  "
            f"ports {one_line(scan['ports'], 100)} ({scan['port_count']:,}, "
            f"{protocols(scan['protocols'])}, {ENGINES.get(scan['engine'], scan['engine'])})  "
            f"hosts: {hosts or 'none'}  open ports: {scan['open_ports']}  worst: {worst(scan)}  "
            f"from {one_line(scan['source'], 60)}")


def scans_list(answer: Mapping[str, Any]) -> None:
    """``--history``: one line per saved scan, newest first; how to see the older ones."""
    scans = answer["scans"]
    if not scans:
        print(NO_SCANS)
    for i, scan in enumerate(scans, 1):
        print(f"{i}. {scan_line(scan)}")
    if answer["left_out"]:
        print(f"⚠️ {answer['left_out']} file(s) in instance/scan_history are not scans this "
              "toolkit can show (edited, or from a newer version): left out.")
    if answer["more"] and scans:
        print(f"Older scans: add --before {scans[-1]['scan_id']}")


def _host_line(host: Mapping[str, Any], reputation: Mapping[str, Any]) -> str:
    """A host: how it ended, its OS hint and AbuseIPDB's answer, or why it failed."""
    line = f"  {host['target']}: {HOST_STATES.get(host['state'], host['state'])}"
    message = host.get("message")
    if host["state"] == "failed" and message:
        line += f" ({one_line(message, 300)})"
    if host["os"]:
        line += f"; OS: {one_line(host['os'], 300)}"
    found = reputation.get(host["target"])
    if found:
        line += ("; AbuseIPDB: the lookup failed (" + one_line(found["error"], 200) + ")"
                 if "error" in found else
                 f"; AbuseIPDB: confidence of abuse {one_line(found.get('abuseConfidenceScore'), 10)}%, "
                 f"{one_line(found.get('totalReports'), 10)} report(s)")
    return line


def again_command(again: Mapping[str, Any]) -> str:
    """The command that scans the same again (its authorization is asked again)."""
    words = ["python -m toolkit scan", one_line(again["target"], 200), "-p",
             one_line(again["ports"], 200), "--protocol", again["protocol"]]
    if again["engine"] == "nmap":
        words += ["--engine nmap --service-detection", again["service_detection"], "--timing",
                  again["timing"]]
        words += ["--scripts", again["scripts"]] if again["scripts"] else []
        words += ["--os-detect"] if again["os_detect"] else []
    words += ["--scan-unresponsive"] if again["scan_unresponsive"] else []
    words += [flag for name, flag in (("nvd", "--no-nvd"), ("ai", "--no-ai"),
                                      ("abuseipdb", "--no-abuseipdb")) if not again[name]]
    return " ".join(words)


def show(scan: Mapping[str, Any], hosts: Sequence[Mapping[str, Any]]) -> None:
    """``--show``: one saved scan (``saved_scans.scan_view``) as a scan prints its results;
    ``hosts`` are the record's, with why a host failed."""
    print(f"Scan {scan['scan_id']}: started {when(scan['started'])}, asked from "
          f"{one_line(scan['source'], 60)}")
    print(f"Target: {one_line(scan['target'], 200)} ({', '.join(h['target'] for h in hosts)})")
    print(f"Ports: {one_line(scan['port_list'], 200)} ({scan['ports']:,} ports, "
          f"{protocols(scan['protocols'])})")
    print(f"Engine: {scan['engine_notice'][0]}")
    print("Online in this scan:")
    for key, name in ONLINE:
        print(f"  {name}: {one_line(scan['online'].get(key, '?'), 200)}")
    print("Hosts:")
    for host in hosts:
        print(_host_line(host, scan["reputation"]))
    many = len({entry["target"] for entry in scan["entries"]}) > 1
    for i, entry in enumerate(scan["entries"]):
        if many and (i == 0 or scan["entries"][i - 1]["target"] != entry["target"]):
            print(f"\n{'=' * 60}\nHost: {entry['target']}")
        print_entry(entry)
    totals = scan["summary"]
    print(f"\n{scan['message']}")
    if totals["failed_lookups"]:
        print("Failed lookups: " + "; ".join(totals["failed_lookups"]))
    print("Requests sent online: " + ", ".join(f"{name} {n}" for name, n in totals["sent"].items()))
    if totals["unavailable"]:
        print("Unavailable: " + "; ".join(f"{name} ({why})"
                                          for name, why in totals["unavailable"].items()))
    print(f"To scan it again: {again_command(scan['again'])}")


def _side(found: Mapping[str, Any] | None) -> str:
    """One scan's side of a port."""
    if found is None:
        return "–"
    severity = (found["severity"] + (" (unconfirmed)" if found["severity_unconfirmed"] else "")
                + ("" if found["max_cvss"] is None else f", highest CVSS {found['max_cvss']}")
                if found["software"] else "no lookup")
    return (f"{one_line(found['service'], 30) or 'no service'} · "
            f"{one_line(found['software'], 60) or 'software not recognised'} · {severity}")


def _cves(row: Mapping[str, Any]) -> str:
    """The CVEs added and gone, or why they weren't compared."""
    if row["cves_added"] is None:
        return row["cves_note"] or "–"
    if not row["cves_added"] and not row["cves_gone"]:
        return "no change"
    return "; ".join(part for part in (
        "added: " + ", ".join(row["cves_added"]) if row["cves_added"] else "",
        "gone: " + ", ".join(row["cves_gone"]) if row["cves_gone"] else "") if part)


def _row(row: Mapping[str, Any]) -> None:
    """One port of a compared host."""
    status = STATUS.get(row["status"], row["status"])
    if row["changes"]:
        status += ": " + ", ".join(CHANGE_NAMES.get(name, name) for name in row["changes"])
    print(f"  {str(row['port']) + '/' + row['protocol']:<8} {status}")
    print(f"      A: {_side(row['a'])}")
    print(f"      B: {_side(row['b'])}")
    print(f"      CVEs: {_cves(row)}")


def comparison(answer: Mapping[str, Any]) -> None:
    """``--compare``: two saved scans compared (``scan_comparison.compare_scans``), A the older."""
    for name, scan in (("A, the older", answer["a"]), ("B, the newer", answer["b"])):
        print(f"{name}: scan {scan['scan_id']}, {when(scan['started'])}, "
              f"{one_line(scan['target'], 100)}, ports {one_line(scan['ports'], 100)} "
              f"({protocols(scan['protocols'])})")
    if not answer["same_target"]:
        print("The two scans had different targets: only the addresses both scanned are "
              "compared.")
    for name in ("a", "b"):
        if answer[f"only_in_{name}"]:
            print(f"Only in {name.upper()}: {', '.join(answer[f'only_in_{name}'])} "
                  "(not compared).")
    if not answer["common"]:
        print("No address in common: there is nothing to compare.")
        return
    t = answer["totals"]
    print(f"{t['opened']} port{'' if t['opened'] == 1 else 's'} opened, {t['closed']} closed, "
          f"{t['changed']} changed, {t['same']} the same"
          + (f", {t['not_scanned']} open in one scan that the other didn't check"
             if t["not_scanned"] else "")
          + f"; CVEs: {t['cves_added']} added, {t['cves_gone']} gone. A port is closed only "
            "when the newer scan checked it, and CVEs are compared only when both scans looked "
            "them up.")
    for host in answer["hosts"]:
        if not host["compared"]:
            print(f"  {host['address']}: not compared: in A it was "
                  f"{HOST_STATES.get(host['a'], host['a'])}, in B "
                  f"{HOST_STATES.get(host['b'], host['b'])}.")
            continue
        print(f"\n{host['address']}:" + ("" if host["rows"] else " no open port in either scan."))
        for row in host["rows"]:
            _row(row)


def delete_notice(scan_id: str) -> None:
    """``--delete``: what goes."""
    print(f"Deleting scan {scan_id} removes its record, {scan_id}.json, from "
          "instance/scan_history on this computer; it can't be undone. A report you exported "
          "is not touched.")


def confirm_delete() -> bool:
    """The user's yes; anything else, or the end of the input, is no."""
    try:
        return input("Delete it? [y/N] ").strip().lower() in YES
    except EOFError:
        return False


def deleted(scan_id: str | None) -> None:
    """After ``--delete``: the scan deleted, or nothing (``scan_id`` None)."""
    print("Nothing deleted." if scan_id is None else f"✅ Deleted scan {scan_id}.")
