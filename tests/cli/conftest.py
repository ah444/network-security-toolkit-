"""Fixtures for the IDS CLI tests."""

import pytest

from toolkit.cli.controllers import ids
from toolkit.core.services.ids import prediction
from toolkit.cli.ui import ids as ids_ui


@pytest.fixture(scope="module")
def cli_module():
    """The IDS command (``toolkit/cli/controllers/ids.py`` since v5.1 step 1.5d)."""
    return ids


@pytest.fixture(scope="module")
def ids_screen():
    """What the IDS command shows and asks (``toolkit/cli/ui/ids.py``)."""
    return ids_ui


@pytest.fixture(scope="module")
def ids_prediction():
    """The predictions behind it (``toolkit/core/services/ids/prediction.py``)."""
    return prediction
