"""B6: OS guess from TTL ranges, not exact values."""

from toolkit.recon import os_detect


def test_ttl_ranges():
    unix = os_detect.guess_os(64)[0]
    assert unix.startswith("Linux")
    assert os_detect.guess_os(52) == (unix, 12)  # 12 router hops away
    assert os_detect.guess_os(128) == ("Windows", 0)
    assert os_detect.guess_os(117) == ("Windows", 11)
    assert os_detect.guess_os(65)[0] == "Windows"
    assert "Cisco" in os_detect.guess_os(255)[0] and os_detect.guess_os(250)[1] == 5
    for impossible in (None, 0, -1, 256):
        assert os_detect.guess_os(impossible) is None
