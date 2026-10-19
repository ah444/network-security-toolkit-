"""Repository hygiene: secrets and datasets must never be committed."""

import shutil
import subprocess

import pytest

DATA_EXTENSIONS = (".csv", ".txt", ".parquet", ".pcap", ".arff")
# Text files that are legitimately tracked.
ALLOWED_TEXT_FILES = {"requirements.txt", "requirements-dev.txt", "api_keys.example.txt"}

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(repo_root, *args):
    return subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, text=True, encoding="utf-8"
    )


def test_instance_dir_is_gitignored(repo_root):
    result = _git(repo_root, "check-ignore", "-q", "instance/signing.key")
    assert result.returncode == 0, "instance/ must be gitignored (signing key lives there)"


def test_no_dataset_files_tracked(repo_root):
    tracked = _git(repo_root, "ls-files").stdout.splitlines()
    offenders = [
        path
        for path in tracked
        if path.lower().endswith(DATA_EXTENSIONS)
        and not path.startswith("tests/fixtures/")
        and path.rsplit("/", 1)[-1] not in ALLOWED_TEXT_FILES
    ]
    assert offenders == [], f"dataset files must not be committed: {offenders}"
