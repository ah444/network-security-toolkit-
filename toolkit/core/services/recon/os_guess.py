"""Guess a host's OS from what the scan saw (B7, step A3): the TTL of its ping reply, words in
its banners and its open ports. Each is a piece of ``Evidence`` that says where it was seen;
``os_hint`` combines them, says when they disagree, and when the target is this computer (whose
ports may lead to a VM, a container or WSL). Only a guess: TTLs can be set, banners can lie."""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ...data.recon.scanner import PortResult

# Common initial TTLs. Each router hop lowers the TTL by one, so a reply's TTL is at most
# its initial value: the guess is the smallest initial TTL that is not below it.
INITIAL_TTLS = (
    (64, "Linux / macOS / BSD / other Unix", "Unix"),
    (128, "Windows", "Windows"),
    (255, "Network device (e.g. Cisco) or Solaris/AIX", "Network device"),
)
UNIX_LIKE = {"Unix", "Linux", "BSD", "macOS"}  # "Unix" (TTL 64) agrees with each of the others


def _word(name: str, also: str = "") -> re.Pattern[str]:
    """``name`` as a whole word, or the pattern ``also``."""
    return re.compile(r"\b" + name + r"\b" + (f"|{also}" if also else ""))


# Words in a banner: (pattern, family, what it says); the first that matches a banner counts
BANNER_WORDS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (_word("Windows", r"OpenSSH_for_Windows|Microsoft-IIS|Microsoft-HTTPAPI|Microsoft FTP|"
                      r"Microsoft ESMTP|\(Win(?:32|64)\)"), "Windows", "Windows"),
    (re.compile(r"ubuntu", re.IGNORECASE), "Linux", "Linux (Ubuntu)"),
    (_word("Debian", r"\+deb\d+u\d"), "Linux", "Linux (Debian)"),
    (_word("CentOS"), "Linux", "Linux (CentOS)"),
    (_word("RHEL", r"Red Hat|\.el[5-9]\b"), "Linux", "Linux (Red Hat)"),
    (_word("Fedora"), "Linux", "Linux (Fedora)"),
    (_word("Alpine"), "Linux", "Linux (Alpine)"),
    (_word("Raspbian"), "Linux", "Linux (Raspbian)"),
    (_word("FreeBSD"), "BSD", "FreeBSD"),
    (_word("OpenBSD"), "BSD", "OpenBSD"),
    (_word("NetBSD"), "BSD", "NetBSD"),
)
# Open TCP ports that only one OS family has: port -> (family, what it says, the service)
PORT_SIGNS = {135: ("Windows", "Windows", "Microsoft RPC")}
CONFLICT = "unsure: the evidence conflicts"
CONFLICT_NOTE = ("The ping may be answered by another device than the services (NAT, a "
                 "forwarded port), or a banner may name software of another system.")
THIS_COMPUTER_NOTE = ("This computer's ports may lead to a VM, a container or WSL (a forwarded "
                      "port): the services that disagree run there.")


@dataclass(frozen=True)
class Evidence:
    """One sign of the OS: its family (to compare signs), what it says, and where it was seen."""

    family: str
    says: str
    seen: str


@dataclass(frozen=True)
class OsHint:
    """What the evidence says about one host's OS: the verdict, the evidence it rests on, and
    why the evidence may disagree ("" when it agrees)."""

    verdict: str
    evidence: tuple[Evidence, ...]
    note: str = ""
    this_computer: bool = False

    def summary(self) -> str:
        """One line for the report: the verdict and each piece of evidence."""
        signs = "; ".join(f"{e.says} ({e.seen})" for e in self.evidence)
        return f"{self.verdict}: {signs}" if signs else self.verdict


def _initial(ttl: int | None) -> tuple[int, str, str] | None:
    """(initial TTL, OS name, family) for a reply's TTL, or None."""
    if ttl is None or not 1 <= ttl <= 255:
        return None
    return next(row for row in INITIAL_TTLS if ttl <= row[0])


def guess_os(ttl: int | None) -> tuple[str, int] | None:
    """(OS family, hops away) from a reply's TTL, or None. A guess only: TTLs can be set."""
    found = _initial(ttl)
    return (found[1], found[0] - ttl) if found and ttl is not None else None


def ttl_evidence(ttl: int | None) -> list[Evidence]:
    """What the ping reply's TTL says (nothing without a TTL)."""
    found = _initial(ttl)
    if found is None or ttl is None:
        return []
    initial, name, family = found
    return [Evidence(family, name, f"TTL {ttl}, about {initial - ttl} hops away")]


def banner_evidence(results: Iterable[PortResult]) -> list[Evidence]:
    """What words in the banners say, one piece per thing said, with the ports that said it."""
    ports: dict[tuple[str, str], list[str]] = {}
    for result in results:
        for pattern, family, says in BANNER_WORDS:
            if pattern.search(result.banner):
                ports.setdefault((family, says), []).append(f"{result.port}/{result.protocol}")
                break
    return [Evidence(family, says, "the banner of " + ", ".join(seen))
            for (family, says), seen in ports.items()]


def port_evidence(results: Iterable[PortResult]) -> list[Evidence]:
    """What the open TCP ports in ``PORT_SIGNS`` say."""
    found = []
    for result in results:
        if result.protocol == "tcp" and result.port in PORT_SIGNS:
            family, says, service = PORT_SIGNS[result.port]
            found.append(Evidence(family, says, f"{result.port}/tcp open, {service}"))
    return found


def agree(first: str, second: str) -> bool:
    """Whether two families can be the same OS ("Unix" from a TTL fits Linux, BSD or macOS)."""
    return first == second or ({first, second} <= UNIX_LIKE and "Unix" in (first, second))


def _family_of(os_name: str) -> str:
    """The family of this computer's OS name ("Windows 10" is "Windows")."""
    return {"Windows": "Windows", "Linux": "Linux", "macOS": "macOS"}.get(
        os_name.split(" ", 1)[0], "")


def _most_specific(evidence: Sequence[Evidence]) -> str:
    """What the agreeing evidence says most precisely: a family over "Unix", a distribution
    ("Linux (Ubuntu)") over a family."""
    specific = [e for e in evidence if e.family != "Unix"] or list(evidence)
    return max(specific, key=lambda e: "(" in e.says).says


def os_hint(ttl: int | None, open_ports: Sequence[PortResult], this_computer: bool = False,
            local_os: str = "") -> OsHint | None:
    """The OS from the TTL, the banners and the open ports: what they agree on, or that they
    conflict; for this computer its own OS (``local_os``), and why some evidence may disagree.
    None without any evidence."""
    evidence = (*ttl_evidence(ttl), *banner_evidence(open_ports), *port_evidence(open_ports))
    if this_computer:
        family = _family_of(local_os)
        odd = bool(family) and any(not agree(e.family, family) for e in evidence)
        return OsHint(f"this computer ({local_os})", evidence, THIS_COMPUTER_NOTE if odd else "",
                      this_computer=True)
    if not evidence:
        return None
    if all(agree(a.family, b.family) for a in evidence for b in evidence):
        return OsHint(_most_specific(evidence), evidence)
    return OsHint(CONFLICT, evidence, CONFLICT_NOTE)
