"""Step 0.3 (S17, B18): the CLI only unpickles files pinned in models/manifest.json."""

import hashlib
import json
import os
import pickle
import shutil

import pytest
from sklearn.exceptions import InconsistentVersionWarning

MODELS = "tools/IDS/IDS_CLI - CIC-2017/models"
SMALL_FILES = ["config.json", "manifest.json", "scaler.pkl", "label_encoder.pkl", "xgboost.pkl"]


class _Evil:
    """Unpickling this would run os.mkdir(marker): proof that a file was deserialized."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (os.mkdir, (self.marker,))


@pytest.fixture
def models_copy(tmp_path, repo_root):
    """A copy of the small v4 files, so tests can add or tamper with files safely."""
    target = tmp_path / "models"
    target.mkdir()
    for name in SMALL_FILES:
        shutil.copy2(repo_root / MODELS / name, target / name)
    return target


def test_manifest_matches_shipped_files(repo_root):
    models = repo_root / MODELS
    manifest = json.loads((models / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((models / name).read_bytes()).hexdigest() == digest, name
    assert manifest["trained_with"]["scikit-learn"] == "1.6.1"


def test_cli_refuses_unpinned_model(cli_module, models_copy, tmp_path, capsys):
    marker = tmp_path / "PWNED"
    (models_copy / "evil.pkl").write_bytes(pickle.dumps(_Evil(str(marker))))
    ids = cli_module.IDSModel(models_dir=str(models_copy))
    assert ids.load("evil") is False
    assert not marker.exists(), "an untrusted pickle was deserialized"
    assert "not trusted" in capsys.readouterr().out.lower()


def test_cli_refuses_tampered_pinned_file(cli_module, models_copy, capsys):
    with (models_copy / "label_encoder.pkl").open("ab") as handle:
        handle.write(b"\x00")
    ids = cli_module.IDSModel(models_dir=str(models_copy))
    assert ids.load("xgboost") is False
    assert "not trusted" in capsys.readouterr().out.lower()


def test_unpinned_models_are_not_listed(cli_module, models_copy):
    (models_copy / "evil.pkl").write_bytes(b"not a model")
    ids = cli_module.IDSModel(models_dir=str(models_copy))
    assert "evil" not in ids.get_available_models()


def test_cli_v4_version_mismatch_warns(cli_module, capsys, monkeypatch):
    monkeypatch.setattr(cli_module, "_installed_version", lambda package: "1.8.0")
    ids = cli_module.IDSModel()
    with pytest.warns(InconsistentVersionWarning):  # no longer hidden (B19)
        assert ids.load("random_forest")
    out = capsys.readouterr().out
    assert "version mismatch" in out.lower()
    assert "1.6.1" in out and "1.8.0" in out
