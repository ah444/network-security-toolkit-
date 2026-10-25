"""Text files the commands read and write (UTF-8)."""

from pathlib import Path


def read_text(path: str | Path) -> str:
    """A text file's content, read as UTF-8."""
    return Path(path).read_text(encoding="utf-8")


def write_new_text(path: str | Path, text: str) -> None:
    """A new file (FileExistsError rather than replace one); line ends exactly as given."""
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)
