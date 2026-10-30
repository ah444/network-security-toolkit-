"""nmap's XML (step N3): one host's ports as ``PortResult``s, its OS matches and host scripts.

The XML is nmap's own (``-oX -``, at most ``nmap_run.MAX_XML_BYTES``). nmap never declares an
entity, so XML that does is refused before it is parsed; ``xml.etree`` reads no external
file. Every text in it that came from the network or is about it (product, version, extra
info, CPE, script output, OS names) is cleaned and cut here (``one_line``, ``clean_text``),
and is still text from the network for whoever shows it.

The states are the built-in scanner's: TCP ``open`` or ``closed`` (nmap's closed, filtered,
...), UDP ``open``, ``open|filtered`` or ``closed``; nmap's own state is kept in ``reason``
when it differs (``filtered (no-response)``). A port nmap counted only in ``<extraports>``
gets that group's state.
"""

from __future__ import annotations

import ipaddress
import xml.etree.ElementTree as ElementTree
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .nmap_run import NmapFailed
from .scanner import PortResult
from .services import over_tls, service_label, usual_name
from .text import clean_text, one_line

STATES = {"tcp": ("open",), "udp": ("open", "open|filtered")}  # kept as nmap says; else closed
NAME_LENGTH, TEXT_LENGTH, OUTPUT_LENGTH = 40, 200, 2000
MAX_SCRIPTS = 20  # per port, and for the host
FAILED_SCRIPT = "ERROR: Script execution failed"  # nmap's line for a script that didn't run
MAX_OS_MATCHES = 3


@dataclass
class NmapHost:
    """What nmap found on one host."""

    ports: list[PortResult]
    os_matches: tuple[tuple[str, int], ...] = ()  # (name, accuracy %), the best first
    scripts: tuple[tuple[str, str], ...] = ()  # the host's scripts: (id, output)


def _ranges(text: str) -> set[int]:
    """``1-3,7`` as {1, 2, 3, 7}; what isn't a number or range is left out."""
    numbers: set[int] = set()
    for part in text.split(","):
        low, _, high = part.partition("-")
        if low.isdigit() and (not high or high.isdigit()):
            numbers.update(range(int(low), int(high or low) + 1))
    return numbers


def _state(protocol: str, nmap_state: str, reason: str) -> tuple[str, str]:
    """(our state, the reason): nmap's state if this scanner has it, else closed with nmap's
    state in the reason."""
    if nmap_state in STATES[protocol]:
        return nmap_state, reason
    if nmap_state == "closed":
        return "closed", reason
    return "closed", f"{nmap_state} ({reason})" if reason else nmap_state


def _scripts(parent: ElementTree.Element | None) -> tuple[tuple[str, str], ...]:
    """The ``<script>`` children that gave an output (a failed one left out), cleaned."""
    if parent is None:
        return ()
    found = []
    for script in parent.findall("script")[:MAX_SCRIPTS]:
        output = clean_text(script.get("output", ""), OUTPUT_LENGTH).strip()
        if output and not output.startswith(FAILED_SCRIPT):
            found.append((one_line(script.get("id", ""), NAME_LENGTH), output))
    return tuple(found)


def describe(product: str, version: str, extrainfo: str) -> str:
    """nmap's description of a service: ``OpenSSH for_Windows_8.1 (protocol 2.0)``."""
    text = " ".join(part for part in (product, version) if part)
    return f"{text} ({extrainfo})" if text and extrainfo else text or extrainfo


def _service_name(protocol: str, port: int, service: ElementTree.Element | None) -> str:
    """The service as the built-in scanner names it: nmap's name, the TLS name over TLS
    (``https``), with a "?" (TCP) when nmap only took it from the port number."""
    if service is None or not service.get("name"):
        return service_label(port, "") if protocol == "tcp" else usual_name(port, protocol)
    name = one_line(service.get("name", ""), NAME_LENGTH)
    if service.get("tunnel") == "ssl":
        name = over_tls(name) if over_tls(name) != name else f"ssl/{name}"
    if service.get("method") == "table" and protocol == "tcp":
        name += "?"
    return name


def _port(element: ElementTree.Element, protocol: str, port: int) -> PortResult:
    """One ``<port>`` element as a PortResult."""
    found = element.find("state")
    state, reason = _state(protocol, one_line(found.get("state", "") if found is not None else "",
                                              NAME_LENGTH),
                           one_line(found.get("reason", "") if found is not None else "",
                                    NAME_LENGTH))
    service = element.find("service")
    get = service.get if service is not None else (lambda key, default="": default)
    product = one_line(get("product", ""), TEXT_LENGTH)
    version = one_line(get("version", ""), TEXT_LENGTH)
    extrainfo = one_line(get("extrainfo", ""), TEXT_LENGTH)
    scripts = _scripts(element)
    banner = next((output for script, output in scripts if script == "banner"), "")
    if "\\x" in banner:  # bytes nmap wrote as \x00: its description says more
        banner = ""
    return PortResult(
        port, protocol, state,
        service=_service_name(protocol, port, service) if state != "closed" else "",
        banner=one_line(banner, TEXT_LENGTH) or describe(product, version, extrainfo),
        product=product, version=version,
        cpe=tuple(one_line(c.text or "", TEXT_LENGTH) for c in
                  (service.findall("cpe") if service is not None else [])
                  if (c.text or "").startswith("cpe:/")),
        reason=reason, scripts=scripts)


def _extra_states(host: ElementTree.Element) -> tuple[dict[tuple[str, int], tuple[str, str]],
                                                       tuple[str, str] | None]:
    """The ports nmap counted in ``<extraports>``: by (protocol, port) when nmap named them
    (``ports=``), and, when there is one group and it names none, its state and reason."""
    by_port: dict[tuple[str, int], tuple[str, str]] = {}
    groups = host.findall("ports/extraports")
    for group in groups:
        for counted in group.findall("extrareasons"):
            for number in _ranges(counted.get("ports", "")):
                by_port[(counted.get("proto", ""), number)] = (group.get("state", ""),
                                                               counted.get("reason", ""))
    only = None
    if len(groups) == 1 and not by_port:
        reasons = groups[0].findall("extrareasons")
        only = (groups[0].get("state", ""),
                reasons[0].get("reason", "") if len(reasons) == 1 else "")
    return by_port, only


def _the_host(root: ElementTree.Element, address: str) -> ElementTree.Element:
    """The ``<host>`` of ``address``; NmapFailed with nmap's reason when there is none."""
    wanted = ipaddress.ip_address(address.split("%")[0])
    for host in root.findall("host"):
        for found in host.findall("address"):
            try:
                if ipaddress.ip_address(found.get("addr", "").split("%")[0]) == wanted:
                    if host.get("timedout") == "true":
                        raise NmapFailed(f"nmap gave up on {address}: it took longer than the "
                                         "host timeout (TOOLKIT_SCAN_MAX_MIN).")
                    return host
            except ValueError:
                continue
    finished = root.find("runstats/finished")
    why = one_line(finished.get("errormsg", ""), TEXT_LENGTH) if finished is not None else ""
    raise NmapFailed(f"nmap's XML has no result for {address}" + (f": {why}" if why else "."))


def parse_nmap(xml: bytes, address: str, ports: Iterable[int],
               protocols: Sequence[str]) -> NmapHost:
    """The ports of ``address`` that were asked for (``ports`` x ``protocols``, TCP first, by
    number), each once, its OS matches and host scripts. NmapFailed for XML that can't be read
    or has no result for the host."""
    if b"<!ENTITY" in xml:
        raise NmapFailed("nmap's XML declares an entity, which nmap never does; not read.")
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as error:
        raise NmapFailed(f"nmap's XML could not be read ({error}).") from None
    if root.tag != "nmaprun":
        raise NmapFailed("nmap's output is not nmap's XML.")
    host = _the_host(root, address)
    wanted = [(protocol, port) for protocol in ("tcp", "udp") if protocol in protocols
              for port in sorted(set(ports))]
    listed = {}
    for element in host.findall("ports/port"):
        key = (element.get("protocol", ""), element.get("portid", ""))
        if key[1].isdigit() and (key[0], int(key[1])) in set(wanted):
            listed[(key[0], int(key[1]))] = element
    by_port, only = _extra_states(host)
    results = []
    for protocol, port in wanted:
        if (protocol, port) in listed:
            results.append(_port(listed[(protocol, port)], protocol, port))
            continue
        nmap_state, reason = by_port.get((protocol, port)) or only or ("closed", "")
        state, reason = _state(protocol, one_line(nmap_state, NAME_LENGTH),
                               one_line(reason, NAME_LENGTH))
        service = usual_name(port, protocol) if state != "closed" else ""
        results.append(PortResult(port, protocol, state, service=service, reason=reason))
    matches = tuple((one_line(match.get("name", ""), TEXT_LENGTH), int(match.get("accuracy", "")))
                    for match in host.findall("os/osmatch")[:MAX_OS_MATCHES]
                    if match.get("accuracy", "").isdigit())
    return NmapHost(results, matches, _scripts(host.find("hostscript")))
