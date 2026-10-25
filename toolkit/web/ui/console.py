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
