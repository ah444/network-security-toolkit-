"""B8: a silent UDP port is open|filtered, not closed."""

import socket
import threading


def test_silent_udp_open_filtered(scanner):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as silent:
        silent.bind(("127.0.0.1", 0))
        result = scanner.scan_udp_port("127.0.0.1", silent.getsockname()[1], timeout=0.5)
    assert (result.protocol, result.state) == ("udp", "open|filtered")


def test_udp_reply_is_open(scanner):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
        server.bind(("127.0.0.1", 0))
        server.settimeout(5)

        def answer():
            _, client = server.recvfrom(1024)
            server.sendto(b"hello from udp", client)

        thread = threading.Thread(target=answer)
        thread.start()
        result = scanner.scan_udp_port("127.0.0.1", server.getsockname()[1])
        thread.join()
    assert (result.state, result.banner) == ("open", "hello from udp")


def test_unreachable_udp_port_is_closed(scanner, free_port):
    # the local network stack answers "port unreachable"
    assert scanner.scan_udp_port("127.0.0.1", free_port).state == "closed"
