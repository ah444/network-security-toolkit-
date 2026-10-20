"""Step 0.4f: signed bundle library [D11, S1, S17, B18]. No web route uses it yet."""

import hashlib
import io
import json
import pickle
import zipfile

import joblib
import numpy as np
import pandas as pd
import pytest

from toolkit import bundle_files, run_files
from toolkit.ids_core import bundle
from toolkit.ids_core.bundle import BundleError, BundleVersionWarning, pack, unpack
from toolkit.ids_core.runs import DatasetRef, TrainSettings, train

KEY = bytes(range(32))
OTHER_KEY = bytes(range(1, 33))
CLASSES = ("BENIGN", "DoS", "PortScan")


def _frame(rows=400, seed=0):
    rng = np.random.default_rng(seed)
    label = rng.choice(CLASSES, size=rows, p=[0.6, 0.25, 0.15])
    shift = pd.Series(label).map({"BENIGN": 0.0, "DoS": 2.0, "PortScan": -2.0}).to_numpy()
    return pd.DataFrame({"a": shift + rng.normal(size=rows), "b": rng.normal(size=rows),
                         "proto": rng.choice(["tcp", "udp"], size=rows), "Label": label})


@pytest.fixture(scope="module")
def run():
    settings = TrainSettings(label="Label", strategy="stratified",
                             models=("decision_tree", "logistic_regression"), workers=1)
    return train(_frame(), settings=settings,
                 source=DatasetRef(name="synthetic.csv", sha256="ab" * 32, rows=400, columns=4))


@pytest.fixture(scope="module")
def data(run):
    return pack(run, KEY)


@pytest.fixture
def no_deserialize(monkeypatch):
    """Fails the test if anything reaches joblib.load."""
    def refuse(*args, **kwargs):
        raise AssertionError("joblib.load ran on an untrusted bundle")
    monkeypatch.setattr(bundle.joblib, "load", refuse)


def _code(excinfo):
    return excinfo.value.code


def _with_versions(monkeypatch, run, **changes):
    """Pack ``run`` as if it had been made with other library versions."""
    real = bundle.installed_versions()
    monkeypatch.setattr(bundle, "installed_versions", lambda: {**real, **changes})
    data = pack(run, KEY)
    monkeypatch.setattr(bundle, "installed_versions", lambda: real)
    return data


def _bump(version, part, by=1):
    numbers = [int(x) for x in version.split(".")[:3]]
    numbers += [0] * (3 - len(numbers))
    numbers[part] += by
    return ".".join(map(str, numbers))


# --- round trip --------------------------------------------------------------------------

def test_roundtrip_signed(run, data):
    loaded = unpack(data, key=KEY)
    assert loaded.trust == "signature"
    assert loaded.sha256 == hashlib.sha256(data).hexdigest()
    assert loaded.warnings == ()
    assert loaded.manifest["format_version"] == bundle.FORMAT_VERSION
    assert loaded.run == json.loads(json.dumps(run.to_dict()))  # the TrainingRun, embedded
    assert loaded.run_id == run.run_id and loaded.selected == run.evaluation.selected
    assert loaded.features == run.evaluation.training.features
    assert set(loaded.classes) == set(CLASSES)
    assert set(loaded.models) == {"decision_tree", "logistic_regression"}
    sample = _frame(rows=50, seed=7)
    for name, fitted in run.evaluation.training.models.items():
        assert list(loaded.models[name].predict(sample)) == list(fitted.predict(sample))
    assert loaded.model().name == run.evaluation.selected


def test_pinned_hash_needs_no_key(data, no_deserialize):
    with pytest.raises(BundleError) as wrong:
        unpack(data, pinned_sha256="0" * 64)
    assert _code(wrong) == "HASH_MISMATCH"
    with pytest.raises(BundleError) as neither:
        unpack(data)
    assert _code(neither) == "NO_TRUST"


def test_pinned_hash_loads(data):
    loaded = unpack(data, pinned_sha256=hashlib.sha256(data).hexdigest())
    assert loaded.trust == "pinned"


# --- rejected before deserializing -------------------------------------------------------

def test_tampered_rejected_before_load(data, no_deserialize):
    header = len(bundle.MAGIC) + 8
    manifest_len = int.from_bytes(data[len(bundle.MAGIC):header], "big")
    places = {"manifest length": header - 1,
              "manifest": header + manifest_len // 2,
              "payload": header + manifest_len + 8 + 10,
              "signature": len(data) - 1}
    for where, position in places.items():
        changed = bytearray(data)
        changed[position] ^= 0x01
        with pytest.raises(BundleError) as excinfo:
            unpack(bytes(changed), key=KEY)
        assert _code(excinfo) in {"BAD_SIGNATURE", "TRUNCATED", "EXTRA_DATA"}, where
    # A manifest edited to hide a version mismatch is still caught by the signature
    start = header
    manifest = json.loads(data[start:start + manifest_len])
    manifest["versions"]["scikit-learn"] = "0.0.1"
    body = json.dumps(manifest).encode()
    forged = (bundle.MAGIC + len(body).to_bytes(8, "big") + body + data[start + manifest_len:])
    with pytest.raises(BundleError) as excinfo:
        unpack(forged, key=KEY)
    assert _code(excinfo) == "BAD_SIGNATURE"


def test_truncated_and_empty_rejected(data, no_deserialize):
    cases = {b"": "EMPTY", data[:5]: "NOT_A_BUNDLE", data[:len(bundle.MAGIC) + 3]: "TRUNCATED",
             data[:200]: "TRUNCATED", data[:-1]: "TRUNCATED", data + b"\0": "EXTRA_DATA"}
    for broken, code in cases.items():
        with pytest.raises(BundleError) as excinfo:
            unpack(broken, key=KEY)
        assert _code(excinfo) == code, (len(broken), code)


def test_wrong_format_rejected(run, no_deserialize):
    as_joblib = io.BytesIO()
    joblib.dump({"model": 1}, as_joblib)
    as_zip = io.BytesIO()
    with zipfile.ZipFile(as_zip, "w") as archive:
        archive.writestr("model.pkl", b"x")
    others = [pickle.dumps(run.evaluation.training.models), as_joblib.getvalue(),
              as_zip.getvalue(), b"a,b,Label\n1,2,BENIGN\n", b"\xff" * 64]
    for other in others:
        with pytest.raises(BundleError) as excinfo:
            unpack(other, key=KEY)
        assert _code(excinfo) == "NOT_A_BUNDLE"
        assert "not a toolkit bundle" in str(excinfo.value)


def test_foreign_key_rejected(data, no_deserialize):
    with pytest.raises(BundleError) as excinfo:
        unpack(data, key=OTHER_KEY)
    assert _code(excinfo) == "BAD_SIGNATURE"
    assert "another install" in str(excinfo.value)


# --- library versions [B18] ----------------------------------------------------------------

def test_sklearn_minor_mismatch_refused(monkeypatch, run):
    installed = bundle.installed_versions()["scikit-learn"]
    older = _bump(installed, 1, by=-1) if int(installed.split(".")[1]) else _bump(installed, 0, -1)
    data = _with_versions(monkeypatch, run, **{"scikit-learn": older})
    monkeypatch.setattr(bundle.joblib, "load", lambda *a, **k: pytest.fail("deserialized"))
    with pytest.raises(BundleError) as excinfo:
        unpack(data, key=KEY)
    assert _code(excinfo) == "VERSION_REFUSED"
    message = str(excinfo.value)
    assert f"made with scikit-learn {older}, you have {installed}" in message
    assert "TrainingRun" in message and run.run_id in message


def test_patch_mismatch_warns(monkeypatch, run):
    installed = bundle.installed_versions()["scikit-learn"]
    data = _with_versions(monkeypatch, run, **{"scikit-learn": _bump(installed, 2, by=7)})
    with pytest.warns(BundleVersionWarning, match="scikit-learn"):
        loaded = unpack(data, key=KEY)
    assert len(loaded.warnings) == 1 and "scikit-learn" in loaded.warnings[0]


def test_numpy_major_refused_minor_warns(monkeypatch, run):
    installed = bundle.installed_versions()["numpy"]
    major = _with_versions(monkeypatch, run, numpy=_bump(installed, 0))
    with pytest.raises(BundleError) as excinfo:
        unpack(major, key=KEY)
    assert _code(excinfo) == "VERSION_REFUSED" and "numpy" in str(excinfo.value)
    minor = _with_versions(monkeypatch, run, numpy=_bump(installed, 1))
    with pytest.warns(BundleVersionWarning, match="numpy"):
        assert unpack(minor, key=KEY).warnings


def test_version_check_only_for_libraries_the_models_use(monkeypatch, run):
    # decision tree + logistic regression: XGBoost and LightGBM are not inside the bundle
    data = _with_versions(monkeypatch, run, xgboost="0.1.0", lightgbm=None)
    assert unpack(data, key=KEY).warnings == ()


def test_check_versions_rules():
    same = {"scikit-learn": "1.8.0", "numpy": "2.3.5", "pandas": "2.3.3", "python": "3.14.3"}
    libraries = ("scikit-learn",)
    assert bundle.check_versions(same, same, libraries) == []
    assert len(bundle.check_versions({**same, "python": "3.13.1"}, same, libraries)) == 1
    with pytest.raises(BundleError):  # needed and not installed
        bundle.check_versions(same, {**same, "scikit-learn": None}, libraries)
    with pytest.raises(BundleError):
        bundle.check_versions({**same, "pandas": "3.0.0"}, same, libraries)
    assert bundle.check_versions({**same, "scikit-learn": "1.8.0.post1"}, same, libraries)


# --- edge code: key in instance/, files ---------------------------------------------------

def test_key_created_once(monkeypatch, tmp_path):
    monkeypatch.setenv("TOOLKIT_INSTANCE_DIR", str(tmp_path))
    assert bundle_files.key_path() == tmp_path / "signing.key"
    with pytest.raises(BundleError) as excinfo:
        bundle_files.signing_key()  # loading never creates one
    assert _code(excinfo) == "NO_KEY"
    first = bundle_files.signing_key(create=True)
    assert len(first) == 32 and bundle_files.key_path().is_file()
    assert bundle_files.signing_key(create=True) == first == bundle_files.signing_key()
    bundle_files.key_path().write_text("not a key\n", encoding="utf-8")
    with pytest.raises(BundleError) as bad:
        bundle_files.signing_key()
    assert _code(bad) == "BAD_KEY"


def test_save_and_load_bundle(monkeypatch, tmp_path, run):
    monkeypatch.setenv("TOOLKIT_INSTANCE_DIR", str(tmp_path))
    saved = bundle_files.save_bundle(run)
    assert saved.path == tmp_path / "bundles" / f"{run.run_id}.bundle"
    assert saved.sha256 == hashlib.sha256(saved.path.read_bytes()).hexdigest()
    assert saved.run.bundle_sha256 == saved.sha256 and run.bundle_sha256 is None
    record = run_files.read_run(run_files.write_run(saved.run))  # caller saves the record
    assert record["bundle_sha256"] == saved.sha256
    with pytest.raises(FileExistsError):
        bundle_files.save_bundle(run)  # never overwritten
    loaded = bundle_files.load_bundle(saved.path)
    assert loaded.sha256 == saved.sha256 and loaded.run["bundle_sha256"] is None
    assert bundle_files.load_bundle(saved.path, pinned_sha256=saved.sha256).trust == "pinned"


def test_missing_key_clear_message(monkeypatch, tmp_path, run, no_deserialize):
    monkeypatch.setenv("TOOLKIT_INSTANCE_DIR", str(tmp_path))
    path = bundle_files.save_bundle(run).path
    bundle_files.key_path().unlink()
    with pytest.raises(BundleError) as excinfo:
        bundle_files.load_bundle(path)
    assert _code(excinfo) == "NO_KEY"
    message = str(excinfo.value)
    assert "signing key" in message and "retrain" in message.lower()
    assert "TrainingRun" in message and run.run_id in message
    # a new key does not bring the old bundles back
    bundle_files.signing_key(create=True)
    with pytest.raises(BundleError) as foreign:
        bundle_files.load_bundle(path)
    assert _code(foreign) == "BAD_SIGNATURE"
