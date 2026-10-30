"""Step N3: nmap's XML as the scanner's results: real XML of this PC (tests/fixtures/README.md)
and made-up edge cases; every text from the network cleaned."""

import pytest

from toolkit.core.data.recon.nmap_run import NmapFailed
from toolkit.core.data.recon.nmap_xml import describe, parse_nmap

CONNECT = "nmap_connect_scripts.xml"  # -sT -sV --script (the 8), 7 ports
EXTRAPORTS = "nmap_extraports.xml"  # -sT -sV, 1-150 + 2222, 3306: 149 closed in <extraports>
SYN_UDP_OS = "nmap_syn_udp_os.xml"  # -sS -sU -sV -O


@pytest.fixture
def real(fixtures_dir):
    return lambda name: (fixtures_dir / name).read_bytes()


def _by_port(host):
    return {(r.protocol, r.port): r for r in host.ports}


def _xml(ports="", extra="", address="10.0.0.5", host_attrs="", runstats=""):
    return (f'<?xml version="1.0"?>\n<!DOCTYPE nmaprun>\n<nmaprun scanner="nmap">'
            f'<host{host_attrs}><status state="up"/><address addr="{address}" addrtype="ipv4"/>'
            f'<ports>{extra}{ports}</ports></host>{runstats}</nmaprun>').encode()


def _port(port, state, reason="syn-ack", service="", protocol="tcp", scripts=""):
    return (f'<port protocol="{protocol}" portid="{port}"><state state="{state}" '
            f'reason="{reason}"/>{service}{scripts}</port>')


def test_the_real_connect_scan_with_scripts(real):
    host = parse_nmap(real(CONNECT), "127.0.0.1", [1, 135, 445, 2222, 3306, 5357, 11211], ["tcp"])
    ports = _by_port(host)
    assert [r.port for r in host.ports] == [1, 135, 445, 2222, 3306, 5357, 11211]
    closed = ports[("tcp", 1)]
    assert (closed.state, closed.reason, closed.service, closed.banner) == (
        "closed", "conn-refused", "", "")
    ssh = ports[("tcp", 2222)]
    assert (ssh.state, ssh.reason, ssh.service) == ("open", "syn-ack", "ssh")
    assert (ssh.product, ssh.version) == ("OpenSSH", "for_Windows_8.1")
    assert ssh.cpe == ("cpe:/a:openbsd:openssh:for_windows_8.1",)
    assert ssh.banner == "SSH-2.0-OpenSSH_for_Windows_8.1", "the banner script's output"
    assert "banner" in dict(ssh.scripts)
    assert not any(output.startswith("ERROR") for r in host.ports for _, output in r.scripts), \
        "a script that failed is left out"
    mariadb = ports[("tcp", 3306)]
    assert (mariadb.service, mariadb.product, mariadb.version) == ("mysql", "MariaDB",
                                                                   "5.5.5-10.6.12")
    assert dict(mariadb.scripts)["banner"].startswith("<\\x00"), "nmap's escapes"
    assert mariadb.banner == "MariaDB 5.5.5-10.6.12", "a banner of bytes: nmap's description"
    assert ports[("tcp", 445)].service == "microsoft-ds?", "only from the port number"
    assert ports[("tcp", 135)].product == "Microsoft Windows RPC"
    http = ports[("tcp", 5357)]
    assert http.service == "http" and dict(http.scripts)["http-server-header"] == \
        "Microsoft-HTTPAPI/2.0"
    assert ports[("tcp", 11211)].service == "memcache?", "nmap matched no probe (its servicefp)"
    assert dict(host.scripts)["smb2-time"].startswith("date: 2026-10-25")
    assert "3.1.1" in dict(host.scripts)["smb-protocols"]


def test_the_real_extraports(real):
    wanted = [*range(1, 151), 2222, 3306]
    host = parse_nmap(real(EXTRAPORTS), "127.0.0.1", wanted, ["tcp"])
    ports = _by_port(host)
    assert len(host.ports) == 152
    assert {r.port for r in host.ports if r.state == "open"} == {135, 2222, 3306}
    assert (ports[("tcp", 80)].state, ports[("tcp", 80)].reason) == ("closed", "conn-refused")


def test_the_real_syn_udp_and_os_scan(real):
    host = parse_nmap(real(SYN_UDP_OS), "127.0.0.1", [1, 2, 3, 4, 5, 135, 2222, 53, 123],
                      ["tcp", "udp"])
    ports = _by_port(host)
    assert [(r.protocol, r.port) for r in host.ports][:2] == [("tcp", 1), ("tcp", 2)]
    assert ports[("tcp", 1)].reason == "reset"
    assert (ports[("udp", 53)].state, ports[("udp", 53)].reason) == ("closed", "port-unreach")
    assert ports[("tcp", 135)].state == "open"
    assert host.os_matches == (("Microsoft Windows 10 1809 - 21H2", 100),)


def test_windows_line_ends_too(real):
    crlf = real(CONNECT).replace(b"\n", b"\r\n")
    assert parse_nmap(crlf, "127.0.0.1", [2222], ["tcp"]).ports[0].product == "OpenSSH"


@pytest.mark.parametrize(("protocol", "nmap_state", "state", "reason"), [
    ("tcp", "open", "open", "syn-ack"),
    ("tcp", "closed", "closed", "syn-ack"),
    ("tcp", "filtered", "closed", "filtered (syn-ack)"),
    ("tcp", "unfiltered", "closed", "unfiltered (syn-ack)"),
    ("tcp", "open|filtered", "closed", "open|filtered (syn-ack)"),
    ("udp", "open", "open", "syn-ack"),
    ("udp", "open|filtered", "open|filtered", "syn-ack"),
    ("udp", "filtered", "closed", "filtered (syn-ack)"),
])
def test_states(protocol, nmap_state, state, reason):
    (result,) = parse_nmap(_xml(_port(53, nmap_state, protocol=protocol)), "10.0.0.5", [53],
                           [protocol]).ports
    assert (result.state, result.reason) == (state, reason)


@pytest.mark.parametrize(("service", "name"), [
    ('<service name="http" tunnel="ssl" method="probed" conf="10"/>', "https"),
    ('<service name="ftp" tunnel="ssl" method="probed" conf="10"/>', "ftps"),
    ('<service name="weird" tunnel="ssl" method="probed" conf="10"/>', "ssl/weird"),
    ('<service name="http" method="table" conf="3"/>', "http?"),
    ("", "http?"),  # no <service>: the built-in scanner's label
])
def test_service_names(service, name):
    (result,) = parse_nmap(_xml(_port(80, "open", service=service)), "10.0.0.5", [80],
                           ["tcp"]).ports
    assert result.service == name


def test_ports_not_listed():
    """A port in a named <extraports> range gets its state; one nmap never mentioned is
    closed; a port nmap gave that wasn't asked for is left out."""
    extra = ('<extraports state="filtered" count="2"><extrareasons reason="no-response" '
             'count="2" proto="tcp" ports="20-21"/></extraports>')
    ports = _by_port(parse_nmap(_xml(_port(9999, "open"), extra), "10.0.0.5", [20, 21, 22],
                                ["tcp"]))
    assert set(ports) == {("tcp", 20), ("tcp", 21), ("tcp", 22)}
    assert (ports[("tcp", 20)].state, ports[("tcp", 20)].reason) == (
        "closed", "filtered (no-response)")
    assert (ports[("tcp", 22)].state, ports[("tcp", 22)].reason) == ("closed", ""), \
        "the group names its ports: 22 is not one of them"


def test_one_group_without_port_numbers_covers_the_rest():
    extra = ('<extraports state="open|filtered" count="2"><extrareasons reason="no-response" '
             'count="2" proto="udp"/></extraports>')
    ports = _by_port(parse_nmap(_xml(extra=extra), "10.0.0.5", [161, 162], ["udp"]))
    assert {r.state for r in ports.values()} == {"open|filtered"}
    assert ports[("udp", 161)].service == "snmp", "the usual name, as the built-in scanner"
    two = extra + extra.replace("open|filtered", "closed")
    assert {r.state for r in parse_nmap(_xml(extra=two), "10.0.0.5", [161], ["udp"]).ports} == {
        "closed"}, "two groups and no port numbers: closed"


def test_network_text_is_cleaned_and_cut():
    """XML 1.0 can't hold ESC or BEL, but C1 controls (U+009B is a terminal's CSI) and new
    lines it can: all are removed, every field is cut."""
    hostile = ('<service name="ssh" product="Open&#x9b;31mSSH&#xa;evil" '
               f'version="{"9" * 500}" extrainfo="a&#x85;b" method="probed" conf="10">'
               '<cpe>cpe:/a:x:y:1</cpe><cpe>javascript:alert(1)</cpe></service>')
    script = f'<script id="banner" output="SSH-2.0-&#x9b;2Jx&#xa;{"y" * 5000}"/>'
    (result,) = parse_nmap(_xml(_port(22, "open", service=hostile, scripts=script)),
                           "10.0.0.5", [22], ["tcp"]).ports
    assert result.product == "Open31mSSH evil" and len(result.version) <= 200
    assert result.cpe == ("cpe:/a:x:y:1",), "only CPE names"
    assert result.banner.startswith("SSH-2.0-2Jx y") and len(result.banner) <= 200
    assert all("\x9b" not in output and len(output) <= 2000 for _, output in result.scripts)


def test_xml_with_a_raw_escape_is_refused():
    with pytest.raises(NmapFailed, match="could not be read"):
        parse_nmap(_xml(_port(22, "open", service='<service name="a&#x1b;b"/>')), "10.0.0.5",
                   [22], ["tcp"])


def test_describe():
    assert describe("OpenSSH", "8.9p1", "protocol 2.0") == "OpenSSH 8.9p1 (protocol 2.0)"
    assert describe("nginx", "", "") == "nginx"
    assert describe("", "", "SSDP/UPnP") == "SSDP/UPnP"
    assert describe("", "", "") == ""


def test_an_ipv6_host():
    xml = _xml(_port(22, "open"), address="fe80::1")
    assert parse_nmap(xml, "FE80:0::1", [22], ["tcp"]).ports[0].state == "open"


@pytest.mark.parametrize(("xml", "message"), [
    (b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><nmaprun>&a;</nmaprun>',
     "declares an entity"),
    (b"<nmaprun><host>", "could not be read"),
    (b"<html></html>", "is not nmap's XML"),
    (_xml(address="10.0.0.6"), "no result for 10.0.0.5."),
    (b'<nmaprun><runstats><finished exit="error" errormsg="Failed to resolve &#x9b;1mx"/>'
     b"</runstats></nmaprun>", "no result for 10.0.0.5: Failed to resolve 1mx"),
    (_xml(host_attrs=' timedout="true"'), "gave up on 10.0.0.5"),
])
def test_xml_that_gives_no_result(xml, message):
    with pytest.raises(NmapFailed, match=message.replace(".", r"\.")):
        parse_nmap(xml, "10.0.0.5", [22], ["tcp"])
