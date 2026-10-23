"""Fixtures for the IDS CLI tests."""

import pytest

from toolkit.ids_cli import cli


@pytest.fixture(scope="module")
def cli_module():
    """The CLI module (``toolkit/ids_cli/cli.py`` since v5.1 step 1.4a)."""
    return cli
