"""B6: OS guess from TTL ranges, not exact values."""


def test_ttl_ranges(scanner):
    unix = scanner.guess_os(64)[0]
    assert unix.startswith("Linux")
    assert scanner.guess_os(52) == (unix, 12)  # 12 router hops away
    assert scanner.guess_os(128) == ("Windows", 0)
    assert scanner.guess_os(117) == ("Windows", 11)
    assert scanner.guess_os(65)[0] == "Windows"
    assert "Cisco" in scanner.guess_os(255)[0] and scanner.guess_os(250)[1] == 5
    for impossible in (None, 0, -1, 256):
        assert scanner.guess_os(impossible) is None
