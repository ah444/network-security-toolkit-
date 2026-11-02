"""B12, S14: UTF-8 export to instance/scans, never overwritten, sanitised."""

import csv
import io
import json

import pytest

from toolkit.core.data.recon import scan_files as export, software
from toolkit.core.services.recon import report
from toolkit.core.data.recon import scanner as recon_scanner
from toolkit.core.data.recon import text as recon_text


def _entry(banner, ai="", match="version"):
    result = recon_scanner.PortResult(8080, "tcp", "open", "http-alt", banner)
    intel = {"cves": [{"id": "CVE-2021-41773", "cvss": 9.8, "cvss_version": "3.1",
                       "severity": "CRITICAL", "match": match}],
             "max_cvss": 9.8, "severity": "CRITICAL", "ai_explanation": ai, "errors": [],
             "severity_unconfirmed": match != "version",
             "cve_source": "version match (cpe:2.3:a:apache:http_server:2.4.49)"}
    return report.make_entry("127.0.0.1", result, software.product_version(banner), intel)


def test_export_utf8_ansi_stripped_no_overwrite(tmp_path):
    banner = "\x1b[2J\x1b[31mServeur Apache/2.4.49 café 日本\x1b]0;owned\x07\x07\r\n"
    entries = [_entry(banner)]
    assert entries[0]["banner"] == "Serveur Apache/2.4.49 café 日本\n"
    for fmt in export.EXPORT_FORMATS:
        first = export.export_results(entries, fmt)
        second = export.export_results(entries, fmt)
        assert first != second and first.parent == second.parent
        assert first.parent == export.scans_dir(), "saved in instance/scans"
        assert first.read_bytes() == second.read_bytes(), "the first file is left as it was"
        data = first.read_bytes()
        assert b"\x1b" not in data and b"\x07" not in data
        text = data.decode("utf-8")  # valid UTF-8
        assert "café 日本" in text and "CVE-2021-41773" in text
    report = json.loads(next(export.scans_dir().glob("*.json")).read_text(encoding="utf-8"))
    assert report[0]["software"] == "Apache 2.4.49" and report[0]["severity"] == "CRITICAL"
    with pytest.raises(ValueError, match="Unknown format"):
        export.export_results(entries, "exe")


def test_formula_cells_neutralised(tmp_path):
    payloads = ["=HYPERLINK(\"http://evil\")", "+1+1", "-2+3", "@SUM(A1)", "\tcmd", "\r=1+1"]
    entries = [_entry(payload, ai=payload) for payload in payloads]
    path = export.export_results(entries, "csv", directory=tmp_path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == export.CSV_HEADER
    for row in rows[1:]:
        for cell in (row[4], row[9]):  # banner and AI explanation
            assert not cell.startswith(recon_text.FORMULA_START), cell
            assert cell.startswith("'")
    assert rows[1][1] == "8080", "numbers stay numbers"


@pytest.mark.parametrize(("match", "label", "severity", "confirmed", "unconfirmed"), [
    ("version", "version match", "CRITICAL", "yes", ""),
    ("name", "name match, unconfirmed", "CRITICAL (unconfirmed)", "no (name match only)",
     "CVE-2021-41773"),
])
def test_every_cve_says_how_it_was_matched(scan_screen, capsys, match, label, severity,
                                           confirmed, unconfirmed):
    entries = [_entry("Apache/2.4.49", match=match)]
    rows = list(csv.reader(io.StringIO(export.report_text(entries, "csv"))))
    row = dict(zip(rows[0], rows[1], strict=True))
    assert row["Severity confirmed"] == confirmed
    assert row["Unconfirmed CVEs (name match)"] == unconfirmed
    assert row["Severity (CVSS)"] == "CRITICAL", "the severity column keeps its values"
    text = export.report_text(entries, "txt")
    assert f"CVE-2021-41773 (9.8, {label})" in text and f"Severity (CVSS): {severity}\n" in text
    data = json.loads(export.report_text(entries, "json"))
    assert data[0]["cves"][0]["match"] == match
    assert data[0]["severity_unconfirmed"] is (match == "name")
    scan_screen.print_entry(entries[0])
    out = capsys.readouterr().out
    assert label in out and f"Severity (from CVSS): {severity}\n" in out


def test_terminal_output_is_cleaned(scan_screen, scanner, capsys):
    port = recon_scanner.PortResult(22, "tcp", "open", "ssh", "SSH-2.0\x1b[8mhidden\x1b[0m\r\n")
    scan_screen.print_table([port], "OPEN PORTS")
    scan_screen.print_entry(_entry("\x1b]52;c;ZXZpbA==\x07Apache/2.4.49"))
    out = capsys.readouterr().out
    assert "\x1b" not in out and "\x07" not in out
