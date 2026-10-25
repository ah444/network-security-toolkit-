"""What the menu shows and asks (``toolkit.controllers.cli.main`` runs it)."""

BANNER = r"""
 _       __       __
| |     / /___   / /_____ ____   ____ ___   ___
| | /| / // _ \ / // ___// __ \ / __ `__ \ / _ \
| |/ |/ //  __// // /__ / /_/ // / / / / //  __/
|__/|__/ \___//_/ \___/ \____//_/ /_/ /_/ \___/

Network Recon and Detection Tool
Your one-stop solution for network recon and intrusion detection using Machine Learning.
Let's secure your network with these tools."""

MENU = ("1. Network Recon", "2. Intrusion Detection (CLI)", "3. Intrusion Detection (Web)",
        "4. Exit")


def show_banner() -> None:
    print(BANNER)


def ask_choice() -> str:
    """The menu, then the user's choice (lower case)."""
    print()
    print("\n".join(MENU))
    return input("Enter your choice [1-4]: ").strip().lower()


def not_a_choice(choice: str) -> None:
    print(f"{choice!r} is not a choice: enter 1, 2, 3 or 4 (q also exits).")


def goodbye(after_interrupt: bool = False) -> None:
    print("\nGoodbye!" if after_interrupt else "Goodbye!")


def stopped() -> None:
    print("\nStopped.")


def bad_settings(problems: list[str]) -> None:
    """Each bad TOOLKIT_* setting, one line each."""
    for problem in problems:
        print(problem)


def could_not_start(command: str, error: OSError) -> None:
    print(f"Could not start {command}: {error}")
