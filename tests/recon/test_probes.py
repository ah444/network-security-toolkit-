"""D13: the probes registry, one entry per port (step 1.1d; changed on purpose in step A2).

EXPECTED was a characterization of the scanner before the registry existed (v5.0), generated
from that code: (reader, what is sent, TLS, UDP payload) per port. Step A2 changed it on
purpose: the "detect" entries (53, 389, 636, 1433, 3389, 5432, 9300, 27017) showed a fixed
text without reading anything; now 3389 and 5432 have their own probes, 636 is read over TLS
like any port, the others get the default (read, else ``GET /``); 9100-9107 (printers) get
nothing sent. Every other decision is still v5.0's.
"""

from toolkit.core.data.recon import probes

DNS = (b"\x00\xff\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00"
       b"\x07version\x04bind\x00\x00\x10\x00\x03")
EXPECTED = {
    21: ("greeting", "", False, b"\r\n"),
    22: ("greeting", "", False, b"\r\n"),
    25: ("greeting", "", False, b"\r\n"),
    53: ("nudge", "", False, DNS),
    80: ("http", "", False, b"\r\n"),
    110: ("greeting", "", False, b"\r\n"),
    123: ("nudge", "", False, b"\x1b\x03\x04\xfa\x00\x01\x00\x00\x00\x00\x00\x00"),
    143: ("greeting", "", False, b"\r\n"),
    161: ("nudge", "", False, b"0&\x02\x01\x00\x04\x06public\xa0\x19\x02\x01\x00\x02\x01\x00"
          b"\x02\x01\x000\x110\x0f\x06\x0b+\x06\x01\x02\x01\x01\x01\x00\x05\x00"),
    443: ("http", "", True, b"\r\n"),
    465: ("greeting", "", True, b"\r\n"),
    587: ("greeting", "", False, b"\r\n"),
    636: ("nudge", "", True, b"\r\n"),
    993: ("greeting", "", True, b"\r\n"),
    995: ("greeting", "", True, b"\r\n"),
    3306: ("mysql", "", False, b"\r\n"),
    3389: ("rdp", "", False, b"\r\n"),
    5432: ("postgresql", "", False, b"\r\n"),
    6379: ("request", b"INFO server\r\n", False, b"\r\n"),
    8000: ("http", "", False, b"\r\n"),
    8008: ("http", "", False, b"\r\n"),
    8080: ("http", "", False, b"\r\n"),
    8443: ("http", "", True, b"\r\n"),
    8888: ("http", "", False, b"\r\n"),
    **{port: ("greeting", "", False, b"\r\n") for port in range(9100, 9108)},
    9200: ("http", "", False, b"\r\n"),
    11211: ("request", b"version\r\n", False, b"\r\n"),
}
DEFAULT = ("nudge", "", False, b"\r\n")
WITHOUT_ENTRY = (1, 389, 1433, 4444, 9300, 27017, 31337)


def _decision(port):
    probe = probes.probe_for(port)
    return (probe.read, probe.send or "", probe.tls, probes.udp_payload(port))


def test_registry_decisions():
    assert {port: _decision(port) for port in EXPECTED} == EXPECTED
    assert set(probes.PROBES) == set(EXPECTED), "nothing added or lost"
    assert all(_decision(port) == DEFAULT for port in WITHOUT_ENTRY)


def test_every_probe_has_a_reader():
    for port, probe in probes.PROBES.items():
        assert probe.read in probes.READERS, port
        assert bool(probe.send) == (probe.read == "request"), port


def test_one_entry_adds_a_probe(monkeypatch):
    class FakeSocket:
        sent = b""

        def sendall(self, data):
            self.sent += data

        def recv(self, limit):
            return b"PONG 1.0\r\n"

    monkeypatch.setitem(probes.PROBES, 7777, probes.Probe("request", send=b"PING\r\n"))
    sock = FakeSocket()
    assert probes.read_banner(sock, "127.0.0.1", 7777) == probes.Reply("PONG 1.0", "")
    assert sock.sent == b"PING\r\n"
