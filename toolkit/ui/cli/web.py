"""What the web trainer's command prints."""

PORT_IN_USE = ("Port {port} is already in use on this computer: the web trainer may already be "
               "running (open {url}), or another program uses it.\nStart on another port: "
               "python -m toolkit web --port {other} (or TOOLKIT_PORT={other}).")


def running(url: str) -> None:
    print(f"IDS web app running at {url}  (press Ctrl+C to stop)")


def port_in_use(port: int, url: str, other: int) -> None:
    print(PORT_IN_USE.format(port=port, url=url, other=other))


def debug_mode() -> None:
    print("WARNING: debug mode is ON (TOOLKIT_DEBUG=1). Development use only.")


def open_in_browser(url: str) -> None:
    print(f"Open {url} in your browser.")


def stopped() -> None:
    print("Web trainer stopped.")


def bad_setting(error: Exception) -> None:
    print(error)


def bad_settings(problems: list[str]) -> None:
    """Each bad TOOLKIT_* setting, one line each (as the command line prints them)."""
    for problem in problems:
        print(problem)
