"""S2, S3, S11: uploaded files can't escape uploads/, are never stale, and aren't kept."""

import io

import pytest

from toolkit.web.security import safe_upload_name
CSV_A = b"f1,f2,Label\n1,2,A\n3,4,B\n"
CSV_B = b"f1,f2,Label\n5,6,A\n7,8,B\n9,9,A\n1,1,B\n"


def _post(client, payload: bytes, filename: str):
    return client.post("/api/v1/dataset", data={"dataset": (io.BytesIO(payload), filename)},
                       content_type="multipart/form-data")


@pytest.fixture
def upload_dir(web_app):
    """The app's upload folder (instance/uploads/ since v5.1 step 1.2a)."""
    return web_app.extensions["toolkit"].upload_dir


def _rows(response):
    return int(response.get_json()["message"].split("Total rows: ")[1].split(".")[0])


def test_path_traversal_contained(client, repo_root, upload_dir):
    uploads = upload_dir
    escape_targets = [uploads.parent / "evil.csv", uploads.parent.parent / "evil.csv",
                      repo_root / "evil.csv"]
    try:
        response = _post(client, CSV_A, "../../evil.csv")
        assert response.status_code == 200, response.get_json()
        assert not any(path.exists() for path in escape_targets), "upload escaped uploads/"
    finally:
        for path in escape_targets:  # clean up if the bug ever comes back
            path.unlink(missing_ok=True)


def test_same_name_upload_not_stale(client):
    first = _post(client, CSV_A, "data.csv")
    second = _post(client, CSV_B, "data.csv")
    assert _rows(first) == 2
    assert _rows(second) == 4, "second upload with the same name must use the new content"


def test_uploaded_file_not_kept_on_disk(client, upload_dir):
    _post(client, CSV_A, "data.csv")
    leftovers = [p.name for p in upload_dir.iterdir()]
    assert leftovers == [], f"uploaded data stayed on disk: {leftovers}"


def test_saved_name_is_sanitised():
    name = safe_upload_name("../../we ird<name>.csv")
    assert "/" not in name and "\\" not in name and ".." not in name
    assert name.endswith(".csv")
    assert name != safe_upload_name("../../we ird<name>.csv"), "names must be unique"


def test_display_name_keeps_the_file_name_without_paths(client):
    response = _post(client, CSV_A, "KDD Test+ v2.csv").get_json()
    assert response["dataset"]["files"] == ["KDD Test+ v2.csv"]  # shown and recorded as chosen
    for path in ("../../evil.csv", "C:/Users/me/evil.csv"):
        assert _post(client, CSV_A, path).get_json()["dataset"]["files"] == ["evil.csv"]
