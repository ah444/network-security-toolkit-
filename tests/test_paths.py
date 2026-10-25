"""v5.2 step 2.3a: the toolkit's real paths, pinned before the code moves.

``tests/test_data_folder.py`` checks the rules with a patched ``REPO_ROOT``; this runs the
unpatched code in a fresh process (``tests/paths_probe.py``), so a module that moves and
computes its folder from ``__file__`` is caught. The same probe runs against an installed wheel
after every commit of step 2.3. The expected values here must not change when the code moves.
"""

import json
import os
import subprocess
import sys

import pytest


@pytest.fixture
def paths(repo_root, tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "TOOLKIT_INSTANCE_DIR"}
    env.update(PYTHONPATH=str(repo_root), LOCALAPPDATA=str(tmp_path / "Local"),
               XDG_DATA_HOME=str(tmp_path / "xdg"))
    done = subprocess.run([sys.executable, str(repo_root / "tests" / "paths_probe.py")],
                          cwd=tmp_path, env=env, capture_output=True, text=True,
                          encoding="utf-8", check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_from_the_project_folder(paths, repo_root):
    assert paths["package"] == str(repo_root / "toolkit")
    assert paths["repo_root"] == str(repo_root)
    assert paths["is_checkout"] is True
    assert paths["instance_dir"] == str(repo_root / "instance")
    assert paths["env_file"] == paths["ENV_FILE"] == str(repo_root / ".env")


def test_settings_refuses_to_run_under_another_name(repo_root, tmp_path):
    """v5.2 step 2.3b: REPO_ROOT counts the dots of the module's name; run as ``__main__``
    the count would be wrong, so settings.py stops instead of guessing a folder."""
    env = {k: v for k, v in os.environ.items() if k != "TOOLKIT_INSTANCE_DIR"}
    env.update(PYTHONPATH=str(repo_root))
    done = subprocess.run([sys.executable, "-m", "toolkit.core.data.settings"], cwd=tmp_path,
                          env=env, capture_output=True, text=True, encoding="utf-8", check=False)
    assert done.returncode != 0
    assert "RuntimeError: __main__: settings.py is part of the toolkit package" in done.stderr


def test_the_web_files_are_found(paths):
    assert paths["templates"] == ["dynamic_ids.html", "home.html"]
    assert {"css/app.css", "js/main.js", "js/theme.js", "js/navbar.js"} <= set(paths["static"])
    assert paths["web_inside_package"] is True
