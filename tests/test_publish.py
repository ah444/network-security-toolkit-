"""v5.3 step HR3a (RD4: IA4): a record or a bundle is published whole or not at all, and never
replaces a file. The bytes go to a temporary file in the same folder, then are published with
``os.rename`` on Windows or a hard link on Linux and macOS (tried here on NTFS too); a crash
of the process between the two leaves no file under the final name."""

import errno
import subprocess
import sys
import textwrap

import pytest

from toolkit.core.data import publish
from toolkit.core.data.ids import bundle_files, run_files


def _left(folder):
    return sorted(path.name for path in folder.iterdir())


@pytest.fixture(params=[False, True], ids=["rename", "hard link"])
def way(request, monkeypatch):
    """Both ways of publishing: Windows' rename and Linux's and macOS's hard link (NTFS has
    hard links too, so both run on every system)."""
    monkeypatch.setattr(publish, "HARD_LINKS", request.param)
    return request.param


def test_a_file_is_published_whole(way, tmp_path):
    path = publish.publish_bytes(tmp_path / "runs" / "a.json", b"x" * 100_000)
    assert path.read_bytes() == b"x" * 100_000
    assert _left(tmp_path / "runs") == ["a.json"], "no temporary file left"


def test_text_keeps_its_line_ends(tmp_path):
    publish.publish_text(tmp_path / "a.csv", "a,b\r\n1,é\n")
    assert (tmp_path / "a.csv").read_bytes() == "a,b\r\n1,é\n".encode("utf-8")


def test_a_name_that_exists_is_never_replaced(way, tmp_path):
    (tmp_path / "a.bundle").write_bytes(b"first")
    with pytest.raises(FileExistsError):
        publish.publish_bytes(tmp_path / "a.bundle", b"second")
    assert (tmp_path / "a.bundle").read_bytes() == b"first" and _left(tmp_path) == ["a.bundle"]


def test_a_name_that_appears_meanwhile_is_never_replaced(way, tmp_path, monkeypatch):
    real = publish._finish

    def another_writer_first(temporary, path):
        path.write_bytes(b"theirs")  # after the check, before the publish
        real(temporary, path)

    monkeypatch.setattr(publish, "_finish", another_writer_first)
    with pytest.raises(FileExistsError):
        publish.publish_bytes(tmp_path / "a.json", b"ours")
    assert (tmp_path / "a.json").read_bytes() == b"theirs" and _left(tmp_path) == ["a.json"]


def test_a_failure_leaves_nothing(way, tmp_path, monkeypatch):
    def full(temporary, path):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(publish, "_finish", full)
    with pytest.raises(OSError, match="No space left"):
        publish.publish_bytes(tmp_path / "a.json", b"data")
    assert _left(tmp_path) == []


def test_a_file_system_without_hard_links_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "HARD_LINKS", True)

    def no_links(source, target):
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(publish.os, "link", no_links)
    with pytest.raises(OSError, match="without hard links") as refused:
        publish.publish_bytes(tmp_path / "a.json", b"data")
    assert "TOOLKIT_INSTANCE_DIR" in str(refused.value) and _left(tmp_path) == []


CRASH = textwrap.dedent("""
    import os, sys
    from toolkit.core.data import publish
    def crash(temporary, path):
        os._exit(9)  # the process dies at once: no finally, no clean-up
    publish._finish = crash
    publish.publish_bytes(sys.argv[1], b"b" * 1_000_000)
""")


def test_a_crash_before_the_publish_leaves_no_file_under_its_name(tmp_path):
    target = tmp_path / "bundles" / "run.bundle"
    done = subprocess.run([sys.executable, "-c", CRASH, str(target)], capture_output=True)
    assert done.returncode == 9
    assert not target.exists(), "never a half-written file under the final name"
    [left] = (tmp_path / "bundles").iterdir()  # the temporary file, in the same folder
    assert left.name.startswith("run.bundle.") and left.suffix == ".tmp"
    assert left.stat().st_size == 1_000_000
    assert bundle_files.stored_bundles(tmp_path / "bundles") == [], "no listing reads it"


def test_records_and_bundles_are_published(tmp_path, monkeypatch):
    published = []
    real = publish._finish
    monkeypatch.setattr(publish, "_finish", lambda temporary, path: (
        published.append(path.name), real(temporary, path)))
    run_files.write_record("run-1", {"run_id": "run-1"}, tmp_path / "runs")
    bundle_files.write_new(tmp_path / "bundles" / "run-1.bundle", b"bundle")
    assert published == ["run-1.json", "run-1.bundle"]
    assert run_files.read_record(tmp_path / "runs" / "run-1.json") == {"run_id": "run-1"}
    assert run_files.list_runs(tmp_path / "runs") == [tmp_path / "runs" / "run-1.json"]
