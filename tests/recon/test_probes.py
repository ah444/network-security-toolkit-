"""D13: the probes registry, one entry per port (step 1.1d).

EXPECTED is a characterization of the scanner before the registry existed (v5.0: the sets
HTTP_PORTS, TLS_PORTS, GREETING_PORTS, REQUESTS, DETECT_ONLY, UDP_PAYLOADS and the MySQL
case), generated from that code: (reader, detail, TLS, UDP payload) per port. The
registry must make exactly the same decisions.
"""

from toolkit.data.recon import probes

EXPECTED = {
    21: ("greeting", "", False, b"\r\n"),
    22: ("greeting", "", False, b"\r\n"),
    25: ("greeting", "", False, b"\r\n"),
    53: ("detect", "DNS", False, b"\x00\xff\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00"
         b"\x07version\x04bind\x00\x00\x10\x00\x03"),
    80: ("http", "", False, b"\r\n"),
    110: ("greeting", "", False, b"\r\n"),
    123: ("nudge", "", False, b"\x1b\x03\x04\xfa\x00\x01\x00\x00\x00\x00\x00\x00"),
    143: ("greeting", "", False, b"\r\n"),
    161: ("nudge", "", False, b"0&\x02\x01\x00\x04\x06public\xa0\x19\x02\x01\x00\x02\x01\x00"
          b"\x02\x01\x000\x110\x0f\x06\x0b+\x06\x01\x02\x01\x01\x01\x00\x05\x00"),
    389: ("detect", "LDAP", False, b"\r\n"),
    443: ("http", "", True, b"\r\n"),
    465: ("greeting", "", True, b"\r\n"),
    587: ("greeting", "", False, b"\r\n"),
    636: ("detect", "LDAPS", True, b"\r\n"),
    993: ("greeting", "", True, b"\r\n"),
    995: ("greeting", "", True, b"\r\n"),
    1433: ("detect", "MSSQL", False, b"\r\n"),
    3306: ("mysql", "", False, b"\r\n"),
    3389: ("detect", "RDP", False, b"\r\n"),
    5432: ("detect", "PostgreSQL", False, b"\r\n"),
    6379: ("request", b"INFO server\r\n", False, b"\r\n"),
    8000: ("http", "", False, b"\r\n"),
    8008: ("http", "", False, b"\r\n"),
    8080: ("http", "", False, b"\r\n"),
    8443: ("http", "", True, b"\r\n"),
    8888: ("http", "", False, b"\r\n"),
    9200: ("http", "", False, b"\r\n"),
    9300: ("detect", "Elasticsearch transport", False, b"\r\n"),
    11211: ("request", b"version\r\n", False, b"\r\n"),
    27017: ("detect", "MongoDB", False, b"\r\n"),
    1: ("nudge", "", False, b"\r\n"),  # ports without an entry
    4444: ("nudge", "", False, b"\r\n"),
    31337: ("nudge", "", False, b"\r\n"),
}


def _decision(port):
    probe = probes.probe_for(port)
    return (probe.read, probe.label or probe.send or "", probe.tls, probes.udp_payload(port))


def test_registry_matches_the_v5_0_decisions():
    assert {port: _decision(port) for port in EXPECTED} == EXPECTED
    assert set(probes.PROBES) == set(EXPECTED) - {1, 4444, 31337}  # nothing added or lost


def test_every_probe_has_a_reader():
    for port, probe in probes.PROBES.items():
        assert probe.read in probes.READERS, port
        assert bool(probe.send) == (probe.read == "request"), port
        assert bool(probe.label) == (probe.read == "detect"), port


def test_one_entry_adds_a_probe(monkeypatch):
    class FakeSocket:
        sent = b""

        def sendall(self, data):
            self.sent += data

        def recv(self, limit):
            return b"PONG 1.0\r\n"

    monkeypatch.setitem(probes.PROBES, 7777, probes.Probe("request", send=b"PING\r\n"))
    sock = FakeSocket()
    assert probes.read_banner(sock, "127.0.0.1", 7777) == "PONG 1.0"
    assert sock.sent == b"PING\r\n"
