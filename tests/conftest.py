"""Shared pytest fixtures for the whole test suite."""

import platform
import sys
from pathlib import Path

import pytest

from platforms import PLATFORMS

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def fake_platform(monkeypatch):
    """Make the code under test believe it runs on another OS.

    Usage: ``fake_platform("linux")``. Patches ``platform.system()`` and
    ``sys.platform`` only; OS-specific code must decide through one of these
    (not ``os.name``, which can't be faked safely because the standard
    library depends on it).
    """

    def _set(name: str) -> None:
        system, sys_platform = PLATFORMS[name]
        monkeypatch.setattr(platform, "system", lambda: system)
        monkeypatch.setattr(sys, "platform", sys_platform)

    return _set
