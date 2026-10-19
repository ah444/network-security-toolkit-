"""Fixtures for the IDS CLI tests."""

import importlib.util
import sys

import pytest

CLI_FILE = "tools/IDS/IDS_CLI - CIC-2017/IDS_with_Machine_Learning.py"


@pytest.fixture(scope="module")
def cli_module(request):
    """Import the CLI script as a module (its folder name contains spaces)."""
    path = request.config.rootpath / CLI_FILE
    spec = importlib.util.spec_from_file_location("ids_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ids_cli_under_test"] = module
    spec.loader.exec_module(module)
    return module
