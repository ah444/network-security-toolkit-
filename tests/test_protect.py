"""v5.3 step HS5 (RD1): ``instance/`` and ``.env`` are limited to this account. On Windows the
folder's ACL inherits nothing from above and allows the current user, SYSTEM and
Administrators only, and every file in it (the signing key) inherits that; on Linux and macOS
the modes are 700 and 600. A path that can't be protected is a warning, never a stop. A backup
of the signing key, the runs and the bundles brings them back on a clean instance.

Every test works in its own temporary folder: no real file's permissions are changed. That
another account can't open the key is checked from the ACL Windows reports, not by signing in
as another account."""

import logging
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from toolkit.core.data import logs, protect
from toolkit.core.data.ids import bundle_files
from toolkit.core.services.ids import exports
from toolkit.core.services.ids.runs import DatasetRef, TrainSettings, train

REAL_APPLY = protect._apply  # conftest turns it off for every other test
WINDOWS = pytest.mark.skipif(sys.platform != "win32", reason="Windows' ACLs")
POSIX = pytest.mark.skipif(sys.platform == "win32", reason="chmod's modes")
EVERYONE = "S-1-1-0"


@pytest.fixture
def real(monkeypatch):
    """The real ``_apply``, with nothing protected yet in this process."""
    monkeypatch.setattr(protect, "_apply", REAL_APPLY)
    monkeypatch.setattr(protect, "_done", {})


def _owners():
    return (protect.current_user_sid(), protect.SYSTEM, protect.ADMINISTRATORS)


def _open_to_everyone(tmp_path):
    """An ``instance/`` with a file in it, under a folder that gives Everyone full control (as
    a drive's root often does), so both inherit it."""
    subprocess.run(["icacls", str(tmp_path), "/grant", f"*{EVERYONE}:(OI)(CI)F"], check=True,
                   capture_output=True)
    folder = tmp_path / "instance"
    (folder / "runs").mkdir(parents=True)
    (folder / "runs" / "old.json").write_text("{}", encoding="utf-8")
    assert EVERYONE in protect.access(folder, _owners())[1], "the test's own starting point"
    return folder


@WINDOWS
def test_the_folder_inherits_nothing_and_allows_this_account_only(real, tmp_path):
    folder = _open_to_everyone(tmp_path)
    assert protect.protect(folder) is None
    assert protect.access(folder, _owners()) == (False, [])
    # a file already in it and a key made after it: only what the folder gives them
    assert protect.access(folder / "runs" / "old.json", _owners())[1] == []
    bundle_files.signing_key(folder / "signing.key", create=True)
    assert protect.access(folder / "signing.key", _owners())[1] == []


@WINDOWS
def test_protecting_again_changes_nothing(real, tmp_path):
    folder = _open_to_everyone(tmp_path)
    protect.protect(folder)
    first = protect.access(folder, _owners())
    protect._done.clear()
    assert protect.protect(folder) is None and protect.access(folder, _owners()) == first


@WINDOWS
def test_env_is_this_accounts_only(real, tmp_path, monkeypatch):
    folder = _open_to_everyone(tmp_path)
    env = tmp_path / ".env"
    env.write_text("ABUSEIPDB_API_KEY=test\n", encoding="utf-8")
    monkeypatch.setattr(protect, "env_file", lambda environ=None: env)
    assert protect.protect_private_data({"TOOLKIT_INSTANCE_DIR": str(folder)}) == []
    assert protect.access(env, _owners()) == (False, [])


@POSIX
def test_the_modes_on_linux_and_macos(real, tmp_path):
    folder = tmp_path / "instance"
    folder.mkdir()
    protect.protect(folder)
    key = folder / "signing.key"
    bundle_files.signing_key(key, create=True)
    assert folder.stat().st_mode & 0o777 == 0o700 and key.stat().st_mode & 0o777 == 0o600


def test_a_path_that_cant_be_protected_is_a_warning(monkeypatch, tmp_path, capsys):
    def refuse(path):
        raise OSError("icacls: This operation is not supported")  # e.g. a FAT drive

    monkeypatch.setattr(protect, "_apply", refuse)
    monkeypatch.setattr(protect, "_done", {})
    folder = tmp_path / "instance"
    logs.setup_logging("protect_test", environ={"TOOLKIT_INSTANCE_DIR": str(folder)})
    try:
        warned = capsys.readouterr().err
        assert f"Could not limit who may open {folder}" in warned
        assert "other accounts on this computer may be able to read it" in warned
        assert (folder / "logs" / "protect_test.log").is_file(), "the tool goes on"
    finally:
        logging.getLogger().handlers.clear()


def test_each_path_is_protected_once_per_process(monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(protect, "_apply", asked.append)
    monkeypatch.setattr(protect, "_done", {})
    for _ in range(3):
        protect.protect_private_data({"TOOLKIT_INSTANCE_DIR": str(tmp_path / "instance")})
    assert asked.count((tmp_path / "instance").absolute()) == 1


def test_the_key_is_protected_before_it_is_made(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(protect, "_apply", lambda path: seen.append(
        (path, (path / "signing.key").exists())))
    monkeypatch.setattr(protect, "_done", {})
    bundle_files.signing_key(tmp_path / "signing.key", create=True)
    assert seen == [(tmp_path.absolute(), False)]


# --- A backup and a restore on a clean instance -------------------------------------------

def _run():
    rng = np.random.default_rng(0)
    label = rng.choice(["BENIGN", "DoS"], size=200)
    frame = pd.DataFrame({"a": (label == "DoS") * 2.0 + rng.normal(size=200),
                          "b": rng.normal(size=200), "Label": label})
    settings = TrainSettings(label="Label", strategy="stratified", models=("decision_tree",),
                             workers=1)
    return train(frame, settings=settings,
                 source=DatasetRef(name="synthetic.csv", sha256="ab" * 32, rows=200, columns=3))


def test_a_restore_on_a_clean_instance(monkeypatch, tmp_path):
    first = tmp_path / "first"
    monkeypatch.setenv("TOOLKIT_INSTANCE_DIR", str(first))
    saved = exports.save_bundle(_run())
    exports.write_run(saved.run)
    backup = tmp_path / "backup"
    backup.mkdir()  # what the user guide says to back up: the key, the runs, the bundles
    shutil.copy2(first / "signing.key", backup)
    shutil.copytree(first / "runs", backup / "runs")
    shutil.copytree(first / "bundles", backup / "bundles")
    shutil.rmtree(first)

    clean = tmp_path / "clean"
    monkeypatch.setenv("TOOLKIT_INSTANCE_DIR", str(clean))
    shutil.copytree(backup, clean)
    [listed] = exports.list_bundles()
    assert listed.path.name == saved.path.name
    assert exports.load_bundle(listed.path).sha256 == saved.sha256
    assert len(list((clean / "runs").glob("*.json"))) == 1
