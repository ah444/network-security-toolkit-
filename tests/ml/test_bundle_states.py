"""v5.3 step HR3d (RD4: IA10): a bundle's state from its header. The format records its
lengths, so a few bytes and the file's size say whether it is whole: Exported, Incomplete
(the export was interrupted), or Verification failed once a use found a bad signature. A
listing never unpickles, never checks the HMAC, and never deletes an incomplete bundle."""

import joblib
import numpy as np
import pandas as pd
import pytest

from toolkit.cli.controllers import ids as cli
from toolkit.core.data.ids import bundle_files
from toolkit.core.services.ids import bundle, exports
from toolkit.core.services.ids.runs import DatasetRef, TrainSettings, train
from web_jobs import train_and_wait


@pytest.fixture(scope="module")
def run():
    rng = np.random.default_rng(0)
    label = rng.choice(["BENIGN", "DoS"], size=200)
    frame = pd.DataFrame({"a": (label == "DoS") * 2.0 + rng.normal(size=200),
                          "b": rng.normal(size=200), "Label": label})
    settings = TrainSettings(label="Label", strategy="stratified", models=("decision_tree",),
                             workers=1)
    return train(frame, settings=settings,
                 source=DatasetRef(name="synthetic.csv", sha256="cd" * 32, rows=200, columns=3))


@pytest.fixture
def saved(run):
    return exports.save_bundle(run)


@pytest.fixture(autouse=True)
def nothing_unpickled(monkeypatch):
    """Listing reads headers only: a joblib.load outside a load is a failure."""
    real = joblib.load

    def guarded(*args, **kwargs):
        if not guarded.allowed:
            raise AssertionError("a listing deserialized a model")
        return real(*args, **kwargs)

    guarded.allowed = False
    monkeypatch.setattr(bundle.joblib, "load", guarded)
    return guarded


def _cut(saved, size, name="cut.bundle"):
    path = saved.path.with_name(name)
    path.write_bytes(saved.path.read_bytes()[:size])
    return path


def _listed(store=None):
    return {info.path.name: info for info in (store or exports.BundleStore()).list_bundles()}


def _payload_field(data):
    body = int.from_bytes(data[len(bundle.MAGIC):bundle.HEADER_BYTES], "big")
    return bundle.HEADER_BYTES + body


def test_a_whole_bundle_is_exported(saved):
    info = _listed()[saved.path.name]
    assert info.state == exports.EXPORTED
    assert info.manifest == bundle.peek_manifest(saved.path.read_bytes())


@pytest.mark.parametrize("where", ["empty", "inside the magic", "inside the header",
                                   "inside the manifest", "inside the payload's length",
                                   "inside the payload", "the signature's last byte"])
def test_a_cut_off_bundle_is_incomplete_and_still_listed(saved, where):
    data = saved.path.read_bytes()
    field = _payload_field(data)
    size = {"empty": 0, "inside the magic": 5, "inside the header": len(bundle.MAGIC) + 3,
            "inside the manifest": field - 10, "inside the payload's length": field + 4,
            "inside the payload": field + 8 + 100, "the signature's last byte": len(data) - 1,
            }[where]
    path = _cut(saved, size)
    info = _listed()[path.name]
    assert info.state == exports.INCOMPLETE and info.size == size
    has_manifest = size >= field + 8
    assert (info.manifest == bundle.peek_manifest(data)) if has_manifest else info.manifest == {}
    assert path.read_bytes() == data[:size], "never changed or deleted on its own"


def test_extra_bytes_are_incomplete_too(saved):
    """The lengths don't match the size, whichever way (IA10)."""
    path = saved.path.with_name("longer.bundle")
    path.write_bytes(saved.path.read_bytes() + b"\0")
    assert _listed()[path.name].state == exports.INCOMPLETE


def test_a_length_past_the_file_is_never_read(saved, monkeypatch):
    path = saved.path.with_name("huge.bundle")
    path.write_bytes(bundle.MAGIC + (2**62).to_bytes(8, "big") + b"{}")
    sizes = []
    real = exports.read_start
    monkeypatch.setattr(exports, "read_start",
                        lambda where, size: sizes.append(size) or real(where, size))
    assert _listed()[path.name].state == exports.INCOMPLETE
    assert max(sizes) <= saved.path.stat().st_size, "no read past the largest file"


def test_other_files_stay_out_of_the_list(saved):
    folder = saved.path.parent
    (folder / "notes.bundle").write_bytes(b"not a bundle at all")
    broken = bytearray(saved.path.read_bytes())
    broken[bundle.HEADER_BYTES] = ord("[")  # the manifest is whole but not JSON
    (folder / "broken.bundle").write_bytes(bytes(broken))
    assert set(_listed()) == {saved.path.name}


def test_a_changed_byte_is_exported_until_used_then_verification_failed(saved, nothing_unpickled):
    data = bytearray(saved.path.read_bytes())
    data[_payload_field(data) + 8 + 50] ^= 0xFF
    changed = saved.path.with_name("changed.bundle")
    changed.write_bytes(bytes(data))
    store = exports.BundleStore()
    assert _listed(store)[changed.name].state == exports.EXPORTED, "no HMAC while listing"
    with pytest.raises(bundle.BundleError) as refused:
        store.load(changed)
    assert refused.value.code == "BAD_SIGNATURE"
    listed = _listed(store)
    assert listed[changed.name].state == exports.VERIFICATION_FAILED
    assert listed[saved.path.name].state == exports.EXPORTED, "only the file that failed"
    nothing_unpickled.allowed = True
    assert store.load(saved.path).sha256 == saved.sha256
    assert changed.is_file()


def test_the_newest_whole_export_is_the_default(saved, monkeypatch, nothing_unpickled):
    data = saved.path.read_bytes()
    newer = data.replace(saved.run.run_id.encode(), b"29991231T235959Z-0000beef", 1)
    cut = saved.path.with_name("29991231T235959Z-0000beef.bundle")
    cut.write_bytes(newer[:_payload_field(newer) + 8 + 10])  # its manifest whole, the newest
    monkeypatch.setattr(cli, "MODEL_STORE", exports.BundleStore())
    nothing_unpickled.allowed = True
    assert cli.load_model().sha256 == saved.sha256


def test_the_cli_list_says_which_are_incomplete(saved, capsys):
    path = _cut(saved, 40, "29991231T235959Z-0000beef.bundle")
    assert cli.main(["--list"]) == 0
    out = capsys.readouterr().out
    line = next(text for text in out.splitlines() if path.stem in text)
    assert "INCOMPLETE: the export was interrupted" in line
    assert "INCOMPLETE" not in next(text for text in out.splitlines() if saved.path.stem in text)


def test_the_cli_names_a_bad_signature_after_a_use(saved):
    from toolkit.cli.ui import ids as screen

    info = exports.BundleInfo(saved.path, 10, {}, exports.VERIFICATION_FAILED)
    assert "VERIFICATION FAILED: its signature does not match" in screen.export_line(info)


def test_an_incomplete_bundle_blocks_its_export_and_says_so(run):
    path = bundle_files.bundles_dir() / f"{run.run_id}.bundle"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bundle.MAGIC)
    with pytest.raises(exports.BundleExists) as blocked:
        exports.save_bundle(run)
    assert isinstance(blocked.value, FileExistsError) and blocked.value.state == exports.INCOMPLETE
    assert path.read_bytes() == bundle.MAGIC, "never deleted on its own"


def test_the_page_says_which_bundle_blocks_the_export(client, upload, fixtures_dir):
    columns = upload(fixtures_dir / "cic_ddos_sample.csv").get_json()["columns"]
    trained = train_and_wait(client, {"features": [c for c in columns if c != "Label"],
                                      "target": "Label", "balancing": "none",
                                      "models": ["decision_tree"]})
    run_id = trained.get_json()["run_id"]
    path = bundle_files.bundles_dir() / f"{run_id}.bundle"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bundle.MAGIC)
    answer = client.post("/api/v1/run/export", json={"name": "web"})
    assert answer.status_code == 409
    message = answer.get_json()["error"]["message"]
    assert message == (f"An incomplete bundle of this run is in the bundles folder: its export "
                       f"was interrupted. Remove {run_id}.bundle from instance/bundles, then "
                       "export again (it is never removed on its own).")
    assert path.read_bytes() == bundle.MAGIC
