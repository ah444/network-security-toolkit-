"""B6: OS guess from TTL ranges, not exact values."""

from toolkit.services.recon import os_guess


def test_ttl_ranges():
    unix = os_guess.guess_os(64)[0]
    assert unix.startswith("Linux")
    assert os_guess.guess_os(52) == (unix, 12)  # 12 router hops away
    assert os_guess.guess_os(128) == ("Windows", 0)
    assert os_guess.guess_os(117) == ("Windows", 11)
    assert os_guess.guess_os(65)[0] == "Windows"
    assert "Cisco" in os_guess.guess_os(255)[0] and os_guess.guess_os(250)[1] == 5
    for impossible in (None, 0, -1, 256):
        assert os_guess.guess_os(impossible) is None
