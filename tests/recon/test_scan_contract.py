"""v5.2 step 2.3a: the scan's contract with the network, pinned before the code moves.

Resolving a host name (DNS) may happen before the authorization question, so the user
confirms the actual address. Nothing else may: no ping, no port check, no AbuseIPDB, NVD or AI
request before the validation, the limits, the authorization and the opt-in. A private address
never goes to AbuseIPDB; NVD and the AI get the product and version only. The order of the
steps is checked, not only the result: every step that could reach the network is recorded in
one list, the HTTP requests at the session (``Intel`` itself is the real one).
"""

import builtins
import json
import socket

import pytest

from toolkit.core.data.recon.scanner import PortResult
from toolkit.core.services.recon.intel import Intel

BANNER = "SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.1"
PRODUCT = "OpenSSH 8.9"
CPE = "cpe:2.3:a:openbsd:openssh:8.9"  # the same product and version, in NVD's naming (A4)
AUTHORIZE = "authorised to scan"


@pytest.fixture
def net(scan_service, scanner, fake_session, no_nvd_wait, monkeypatch):
    """Every step that could reach the network, in order: ``dns``, ``authorize``, ``ping``,
    ``scan``, ``http`` (with what was sent). Answers come from ``answers`` (prompt prefix →
    answer); any other question fails the test."""
    state = {"events": [], "alive": True, "keys": {"openrouter": "", "abuseipdb": "", "nvd": ""},
             "answers": {AUTHORIZE: "yes"}, "session": None,
             "results": [PortResult(22, "tcp", "open", "ssh", BANNER)]}
    events = state["events"]

    def getaddrinfo(host, port, *args, **kwargs):
        events.append(("dns", host))
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0))]

    def ask(prompt=""):
        for start, answer in state["answers"].items():
            if start in prompt:
                if start == AUTHORIZE:
                    events.append(("authorize", prompt))
                return answer
        raise AssertionError(f"asked {prompt!r}")

    def ping(host, timeout=1):
        events.append(("ping", host))
        return state["alive"], 64

    def scan(host, ports, protocols, progress=None):
        events.append(("scan", host))
        return list(state["results"])

    class Recorded(fake_session().__class__):
        def _answer(self, method, url, **kwargs):
            events.append(("http", url, json.dumps(kwargs, default=str)))
            return super()._answer(method, url, **kwargs)

    def intel(keys, use_ai=False):
        state["session"] = Recorded(get={}, post={"choices": [{"message": {"content": "x"}}]})
        return Intel(keys, use_ai, session=state["session"])

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(builtins, "input", ask)
    monkeypatch.setattr(scan_service, "ping_host", ping)
    monkeypatch.setattr(scan_service, "scan_ports", scan)
    monkeypatch.setattr(scan_service, "Intel", intel)
    monkeypatch.setattr(scanner, "load_api_keys", lambda: state["keys"])
    return state


def _kinds(events):
    return [event[0] for event in events]


def _http(events, word):
    return [event for event in events if event[0] == "http" and word in event[1]]


def test_the_steps_in_order(scanner, net):
    net["keys"].update(abuseipdb="ab-key", openrouter="or-key")
    assert scanner.main(["8.8.8.8", "-p", "22", "--abuseipdb", "--ai"]) == 0
    events = net["events"]
    kinds = _kinds(events)
    assert kinds[0] == "authorize", "nothing before the authorization (an address: no DNS)"
    abuse = events.index(_http(events, "abuseipdb")[0])
    nvd = events.index(_http(events, "nvd.nist.gov")[0])
    ai = events.index(_http(events, "openrouter")[0])
    assert 0 < abuse < kinds.index("ping") < kinds.index("scan") < min(nvd, ai)


def test_only_dns_before_the_authorization(scanner, net, capsys):
    net["answers"][AUTHORIZE] = "no"
    net["keys"].update(abuseipdb="ab-key", openrouter="or-key")
    assert scanner.main(["scanme.example", "-p", "22", "--abuseipdb", "--ai"]) == 2
    assert _kinds(net["events"]) == ["dns", "authorize"]
    assert net["events"][0] == ("dns", "scanme.example")
    # the address found is shown on the line before the question (which names the host)
    assert "scanme.example resolved to IP: 8.8.8.8" in capsys.readouterr().out


def test_interactive_refusal_sends_nothing(scanner, net):
    net["keys"].update(abuseipdb="ab-key", openrouter="or-key")
    net["answers"] = {"Enter target": "scanme.example", "Enter start": "22", "Enter end": "22",
                      "Select protocol": "tcp", AUTHORIZE: "no"}
    assert scanner.main([]) == 2
    assert _kinds(net["events"]) == ["dns", "authorize"]


@pytest.mark.parametrize("address", ["10.0.0.5", "192.168.1.10", "172.16.0.1", "127.0.0.1",
                                     "169.254.1.1", "100.64.0.1", "fd00::1", "::1"])
def test_a_private_address_never_goes_to_abuseipdb(scanner, net, address):
    net["keys"].update(abuseipdb="ab-key")
    assert scanner.main([address, "-p", "22", "--abuseipdb"]) == 0
    assert _http(net["events"], "abuseipdb") == []
    assert "scan" in _kinds(net["events"])


def test_a_private_address_is_not_even_asked_about(scanner, net):
    net["keys"].update(abuseipdb="ab-key")
    net["answers"] = {"Enter target": "192.168.1.10", "Enter start": "22", "Enter end": "22",
                      "Select protocol": "tcp", AUTHORIZE: "yes", "verbose": "n",
                      "export the results": "n"}
    assert scanner.main([]) == 0
    assert _http(net["events"], "abuseipdb") == []


@pytest.mark.parametrize("key, flags", [("", ["--abuseipdb"]), ("ab-key", [])])
def test_a_public_address_needs_the_key_and_the_flag(scanner, net, key, flags):
    net["keys"].update(abuseipdb=key)
    assert scanner.main(["8.8.8.8", "-p", "22", *flags]) == 0
    assert _http(net["events"], "abuseipdb") == []


def test_a_public_address_needs_the_answer_yes(scanner, net):
    net["keys"].update(abuseipdb="ab-key")
    net["answers"] = {"Enter target": "8.8.8.8", "Enter start": "22", "Enter end": "22",
                      "Select protocol": "tcp", AUTHORIZE: "yes", "verbose": "n",
                      "to AbuseIPDB": "n", "export the results": "n"}
    assert scanner.main([]) == 0
    assert _http(net["events"], "abuseipdb") == []
    net["answers"]["to AbuseIPDB"] = "y"
    net["events"].clear()
    assert scanner.main([]) == 0
    assert len(_http(net["events"], "abuseipdb")) == 1


def test_a_host_that_does_not_answer_is_not_scanned(scanner, net):
    net["alive"] = False
    assert scanner.main(["10.0.0.9", "-p", "22"]) == 0
    assert "scan" not in _kinds(net["events"]) and _http(net["events"], "") == []
    net["answers"] = {"Enter target": "10.0.0.9", "Enter start": "22", "Enter end": "22",
                      "Select protocol": "tcp", AUTHORIZE: "yes", "verbose": "n",
                      "scan this host anyway": "n"}
    net["events"].clear()
    assert scanner.main([]) == 0
    assert "scan" not in _kinds(net["events"]) and _http(net["events"], "") == []
    net["answers"]["scan this host anyway"] = "y"
    net["answers"]["export the results"] = "n"
    net["events"].clear()
    assert scanner.main([]) == 0
    assert "scan" in _kinds(net["events"])


def test_lookups_carry_the_product_only(scanner, net):
    net["keys"].update(openrouter="or-key", nvd="nvd-key")
    assert scanner.main(["scanme.example", "-p", "22", "--ai"]) == 0
    nvd = _http(net["events"], "nvd.nist.gov")
    ai = _http(net["events"], "openrouter")
    # NVD: the version match (the product's CPE name), then, as it found none, the keyword search
    assert [len(nvd), len(ai)] == [2, 1]
    assert CPE in nvd[0][2] and PRODUCT in nvd[1][2] and PRODUCT in ai[0][2]
    for _, url, request in nvd + ai:
        for secret in ("8.8.8.8", "scanme.example", "SSH-2.0", "Ubuntu", "OpenSSH_8.9p1"):
            assert secret not in url + request, (secret, url)


# --- the other steps of scan_target (branches no other test reached) -----------------

def test_a_public_address_without_a_key_is_said_to_be_skipped(scanner, net, capsys):
    net["answers"] = {"Enter target": "8.8.8.8", "Enter start": "22", "Enter end": "22",
                      "Select protocol": "tcp", AUTHORIZE: "yes", "verbose": "n",
                      "export the results": "n"}
    assert scanner.main([]) == 0
    assert "8.8.8.8 is a public IP; AbuseIPDB check skipped (no key)." in capsys.readouterr().out
    assert _http(net["events"], "abuseipdb") == []


def test_udp_ports_with_no_reply_are_listed_and_not_looked_up(scanner, net, capsys):
    net["results"] = [PortResult(53, "udp", "open|filtered"), PortResult(161, "udp", "closed")]
    assert scanner.main(["10.0.0.9", "-p", "53,161", "--protocol", "udp"]) == 0
    out = capsys.readouterr().out
    assert "UDP ports with no reply (open|filtered): 1 (53)" in out
    assert "No open ports found on 10.0.0.9" in out
    assert _http(net["events"], "") == []
