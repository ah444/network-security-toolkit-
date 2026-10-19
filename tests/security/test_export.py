"""S4: export names are restricted and exports never overwrite (active again from 0.4g)."""

import pytest


@pytest.mark.parametrize("bad", ["../../evil", "a/b", r"a\b", "", "   ", "x" * 81, "na:me", ".."])
def test_bad_export_name_refused(app_module, bad):
    with pytest.raises(ValueError):
        app_module.safe_model_basename(bad)


@pytest.mark.parametrize("good", ["my_model", "IDS-v5", "run42"])
def test_good_export_name_kept(app_module, good):
    assert app_module.safe_model_basename(good) == good


def test_export_never_overwrites(app_module, tmp_path):
    first = app_module.unique_export_path(tmp_path, "model", "random_forest")
    first.write_bytes(b"1")
    second = app_module.unique_export_path(tmp_path, "model", "random_forest")
    second.write_bytes(b"2")
    third = app_module.unique_export_path(tmp_path, "model", "random_forest")
    assert first.name == "model_random_forest.pkl"
    assert second.name == "model_random_forest-1.pkl"
    assert third.name == "model_random_forest-2.pkl"
    assert first.read_bytes() == b"1"
    assert third.resolve().parent == tmp_path.resolve()
