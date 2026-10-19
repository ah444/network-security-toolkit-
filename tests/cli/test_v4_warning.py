"""Step 0.2: the CLI must warn that the v4 model uses a label-derived feature (L1)."""

import pytest


def test_warning_text_for_label_derived_feature(cli_module):
    message = cli_module.leakage_warning(["Bwd Packet Length Mean", "Attempted Category"])
    assert message is not None
    assert "Attempted Category" in message
    assert "not reliable" in message.lower()


def test_no_warning_without_label_derived_feature(cli_module):
    assert cli_module.leakage_warning(["Bwd Packet Length Mean", "Packet Length Std"]) is None


@pytest.mark.parametrize("model_name", ["random_forest", "xgboost"])
def test_leaky_model_warning_shown(cli_module, capsys, model_name):
    ids = cli_module.IDSModel()
    assert ids.load(model_name)
    out = capsys.readouterr().out
    assert "Attempted Category" in out
    assert "not reliable" in out.lower()
