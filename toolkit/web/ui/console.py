"""What the web trainer's command prints (``toolkit/ui/cli/web.py`` until v5.2 step 2.3d)."""

PORT_IN_USE = ("Port {port} is already in use on this computer: the web trainer may already be "
               "running (open {url}), or another program uses it.\nStart on another port: "
               "python -m toolkit web --port {other} (or TOOLKIT_PORT={other}).")


def running(url: str) -> None:
    """The address the trainer is served at, and how to stop it."""
    print(f"IDS web app running at {url}  (press Ctrl+C to stop)")


def port_in_use(port: int, url: str, other: int) -> None:
    """The port is taken: what may be using it and how to start on another."""
    print(PORT_IN_USE.format(port=port, url=url, other=other))


def open_to_network(urls: list[str], token: str) -> None:
    """--host (v5.2 step 2.4f): the warning, where other computers open the toolkit, and the
    access token they type (shown here only, never logged)."""
    line = "=" * 72
    print(line)
    print("WARNING: the toolkit is open to your network, not only to this computer.")
    print("Anyone who can reach it and has the access token can upload data, train models")
    print("and run port scans FROM THIS COMPUTER: a holder of the token has your rights over")
    print("the saved runs, bundles and scans, not only a view of the pages. A new token, made")
    print("on this computer's home page, locks every other computer out at once. The")
    print("connection is plain HTTP: someone who can read your network's traffic could")
    print("take over a session. Use it only on a network you trust (such as your virtual")
    print("machines' host-only network), and stop it with Ctrl+C when you are done.")
    print()
    print("Other computers open:" if urls else
          "Other computers open this computer's address on the network, port as above.")
    for url in urls:
        print(f"  {url}")
    print(f"Access token (typed once in each of their browsers): {token}")
    print("This computer's own browser, at 127.0.0.1, needs no token.")
    print(line)


def new_token(token: str) -> None:
    """HS1: the access token made on this computer's page (shown here only, never logged)."""
    line = "=" * 72
    print(line)
    print("A new access token was made: every other computer is locked out until it types")
    print(f"the new one. New access token: {token}")
    print(line)


def not_this_computer(host: str) -> None:
    """--host named an address this computer doesn't have."""
    print(f"{host} is not an address of this computer, so the toolkit can't listen on it. "
          "Use one of its own addresses (ipconfig on Windows, ip addr on Linux, ifconfig on "
          "macOS), or 0.0.0.0 for all of them.")


def debug_refused(host: str) -> None:
    """TOOLKIT_DEBUG=1 with --host: Flask's debugger can run code, so not on a network."""
    print(f"TOOLKIT_DEBUG=1 can't be used with --host {host}: the debug server's debugger "
          "can run any code on this computer. Unset TOOLKIT_DEBUG, or start without --host.")


def debug_mode() -> None:
    """Warns that debug mode is on (for development only)."""
    print("WARNING: debug mode is ON (TOOLKIT_DEBUG=1). Development use only.")


def open_in_browser(url: str) -> None:
    """No browser could be opened: the address to open instead."""
    print(f"Open {url} in your browser.")


def stopped() -> None:
    """The server stopped."""
    print("Web trainer stopped.")


def bad_setting(error: Exception) -> None:
    """A bad setting's message (the server is not started)."""
    print(error)


def bad_settings(problems: list[str]) -> None:
    """Each bad TOOLKIT_* setting, one line each (as the command line prints them)."""
    for problem in problems:
        print(problem)
