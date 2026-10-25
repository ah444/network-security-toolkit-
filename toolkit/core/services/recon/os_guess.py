"""Guess a host's OS family from the TTL of its ping reply (B7). Only a guess:
TTLs can be changed."""

# Common initial TTLs. Each router hop lowers the TTL by one, so a reply's TTL is at most
# its initial value: the guess is the smallest initial TTL that is not below it.
INITIAL_TTLS = (
    (64, "Linux / macOS / BSD / other Unix"),
    (128, "Windows"),
    (255, "Network device (e.g. Cisco) or Solaris/AIX"),
)


def guess_os(ttl: int | None) -> tuple[str, int] | None:
    """(OS family, hops away) from a reply's TTL, or None. A guess only: TTLs can be set."""
    if ttl is None or not 1 <= ttl <= 255:
        return None
    for initial, name in INITIAL_TTLS:
        if ttl <= initial:
            return name, initial - ttl
    return None
