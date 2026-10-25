"""Where the toolkit keeps its data (v5.1 step 1.5b, plan D18).

From the project folder (a clone, or ``pip install -e .``): ``instance/`` and ``.env`` there,
as before. An installed copy (``pip install .``: no pyproject.toml beside its ``toolkit/``)
uses a per-user folder of its OS instead. ``TOOLKIT_INSTANCE_DIR`` always comes first.
"""

from pathlib import Path

import pytest

from toolkit.core.data import settings as config

APP = "network-security-toolkit"


@pytest.fixture
def installed(monkeypatch, tmp_path):
    """toolkit as if installed: its parent folder (site-packages) has no pyproject.toml."""
    site_packages = tmp_path / "site-packages"
    (site_packages / "toolkit").mkdir(parents=True)
    monkeypatch.setattr(config, "REPO_ROOT", site_packages)
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    for name in ("TOOLKIT_INSTANCE_DIR", "LOCALAPPDATA", "XDG_DATA_HOME"):
        monkeypatch.delenv(name, raising=False)
    return home


def test_project_folder_keeps_instance_and_env_beside_the_code(monkeypatch, repo_root):
    monkeypatch.delenv("TOOLKIT_INSTANCE_DIR", raising=False)
    assert config.is_checkout()
    assert config.instance_dir() == repo_root / "instance"
    assert config.env_file() == repo_root / ".env"
    assert config.ENV_FILE == repo_root / ".env"


def test_installed_copy_on_windows_uses_localappdata(installed, fake_platform, monkeypatch,
                                                     tmp_path):
    fake_platform("windows")
    assert not config.is_checkout()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    assert config.user_data_dir() == tmp_path / "Local" / APP
    assert config.instance_dir() == tmp_path / "Local" / APP / "instance"
    assert config.env_file() == tmp_path / "Local" / APP / ".env"
    monkeypatch.delenv("LOCALAPPDATA")
    assert config.user_data_dir() == installed / "AppData" / "Local" / APP


def test_installed_copy_on_macos_uses_application_support(installed, fake_platform):
    fake_platform("darwin")
    assert config.instance_dir() == installed / "Library" / "Application Support" / APP / \
        "instance"


def test_installed_copy_on_linux_follows_xdg(installed, fake_platform, monkeypatch, tmp_path):
    fake_platform("linux")
    assert config.instance_dir() == installed / ".local" / "share" / APP / "instance"
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert config.user_data_dir() == tmp_path / "xdg" / APP
    monkeypatch.setenv("XDG_DATA_HOME", "relative/path")  # ignored, as the XDG spec says
    assert config.user_data_dir() == installed / ".local" / "share" / APP


def test_toolkit_instance_dir_wins_for_an_installed_copy(installed, monkeypatch, tmp_path):
    monkeypatch.setenv("TOOLKIT_INSTANCE_DIR", str(tmp_path / "mine"))
    assert config.instance_dir() == tmp_path / "mine"
