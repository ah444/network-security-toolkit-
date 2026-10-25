"""Fixtures for the IDS CLI tests."""

import pytest

from toolkit.controllers.cli import ids
from toolkit.services.ids import prediction
from toolkit.ui.cli import ids as ids_ui


@pytest.fixture(scope="module")
def cli_module():
    """The IDS command (``toolkit/controllers/cli/ids.py`` since v5.1 step 1.5d)."""
    return ids


@pytest.fixture(scope="module")
def ids_screen():
    """What the IDS command shows and asks (``toolkit/ui/cli/ids.py``)."""
    return ids_ui


@pytest.fixture(scope="module")
def ids_prediction():
    """The predictions behind it (``toolkit/services/ids/prediction.py``)."""
    return prediction
