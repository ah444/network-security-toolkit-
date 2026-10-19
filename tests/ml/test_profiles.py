"""Step 0.4a: content-based dataset profiles [F4, L10, D9]."""

import json
import random

import pytest

from toolkit.ids_core import profiles
from toolkit.ids_core.datasets import inspect_dataset, load_dataset

REAL_FIXTURES = ["cic2017_original_sample.csv", "cic_ddos_sample.csv",
                 "nslkdd_train_sample.txt", "nslkdd_test_sample_noheader.txt"]


def _signature(match):
    return (match.profile.name, match.variant, match.has_header, match.column_names,
            match.label_column)


def _variants(source, tmp_path):
    """The same content under another name, shuffled, with CRLF and other encodings."""
    text = source.read_text(encoding="utf-8")
    lines = text.splitlines()
    has_header = not lines[0][:1].isdigit()
    head, body = (lines[:1], lines[1:]) if has_header else ([], lines)
    shuffled = body[:]
    random.Random(7).shuffle(shuffled)
    yield tmp_path / "renamed.dat", text, "utf-8"
    yield tmp_path / "shuffled.csv", "\n".join(head + shuffled) + "\n", "utf-8"
    yield tmp_path / "crlf.csv", "\r\n".join(lines) + "\r\n", "utf-8"
    yield tmp_path / "bom.csv", text, "utf-8-sig"
    yield tmp_path / "utf16.csv", text, "utf-16"


@pytest.mark.parametrize("fixture", REAL_FIXTURES)
def test_profile_independent_of_filename_order_line_endings(fixtures_dir, tmp_path, fixture):
    expected = _signature(inspect_dataset(fixtures_dir / fixture).profile)
    for path, text, encoding in _variants(fixtures_dir / fixture, tmp_path):
        path.write_bytes(text.encode(encoding))
        assert _signature(inspect_dataset(path).profile) == expected, path.name


def test_cic_variants_detected(fixtures_dir, tmp_path, repo_root):
    full = inspect_dataset(fixtures_dir / "cic2017_original_sample.csv").profile
    assert (full.profile.name, full.variant) == ("cic_ids2017", "original")
    assert full.label_column == "Label"

    subset = inspect_dataset(fixtures_dir / "cic_ddos_sample.csv").profile
    assert (subset.profile.name, subset.variant) == ("cic_ids2017", "original_subset")

    # "Improved" version: header built only from real names, the 8 features of the v4
    # model (trained on it). No header is reconstructed from memory (fixtures README).
    config = json.loads((repo_root / "tools/IDS/IDS_CLI - CIC-2017/models/config.json")
                        .read_text(encoding="utf-8"))
    header = config["features"] + ["Label"]
    path = tmp_path / "improved.csv"
    path.write_text(",".join(header) + "\n" + ",".join(["0"] * 8 + ["BENIGN"]) + "\n",
                    encoding="utf-8")
    improved = inspect_dataset(path).profile
    assert (improved.profile.name, improved.variant) == ("cic_ids2017", "improved")
    assert "Attempted Category" in improved.blocked_columns


def test_cic_split_strategy_follows_time_column(fixtures_dir, tmp_path):
    no_time = inspect_dataset(fixtures_dir / "cic2017_original_sample.csv").profile
    assert no_time.split_strategy == "stratified"
    assert no_time.time_column is None
    assert "CIC_NO_TIME_COLUMN" in no_time.warnings

    source = (fixtures_dir / "cic_ddos_sample.csv").read_text(encoding="utf-8").splitlines()
    rows = [source[0] + ", Timestamp"] + [line + ",2017-07-07 15:00:00" for line in source[1:]]
    path = tmp_path / "with_time.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    timed = inspect_dataset(path).profile
    assert timed.split_strategy == "temporal_per_class"
    assert timed.time_column == "Timestamp"
    assert "Timestamp" in timed.blocked_columns


def test_kddtest_headerless_gets_names(fixtures_dir):
    train = inspect_dataset(fixtures_dir / "nslkdd_train_sample.txt").profile
    test = inspect_dataset(fixtures_dir / "nslkdd_test_sample_noheader.txt").profile
    for match in (train, test):
        assert match.profile.name == "nsl_kdd"
        assert match.label_column == "attack"
        assert match.blocked_columns == ("level",)
        assert match.split_strategy == "official"
    assert train.has_header is True
    assert test.has_header is False
    assert test.column_names == train.column_names
    assert len(test.column_names) == 43
    assert "HEADERLESS_NAMES_APPLIED" in test.warnings


def test_unknown_csv_generic_with_warning(fixtures_dir, tmp_path):
    leak = inspect_dataset(fixtures_dir / "label_leak.csv").profile
    assert leak.profile.name == "generic"
    assert "UNKNOWN_SCHEMA" in leak.warnings
    assert leak.label_column == "Label"
    assert "LABEL_COLUMN_BY_NAME" in leak.warnings
    assert leak.split_strategy == "stratified"

    xss = inspect_dataset(fixtures_dir / "xss_header.csv").profile
    assert xss.profile.name == "generic" and xss.has_header

    path = tmp_path / "numbers.csv"
    path.write_text("1,2,3\n4,5,6\n7,8,9\n", encoding="utf-8")
    bare = inspect_dataset(path).profile
    assert bare.profile.name == "generic"
    assert bare.has_header is False
    assert bare.column_names == ("col_1", "col_2", "col_3")
    assert bare.label_column is None
    assert {"UNKNOWN_SCHEMA", "NO_HEADER", "LABEL_COLUMN_UNKNOWN"} <= set(bare.warnings)
    assert len(load_dataset(path).frame) == 3


def test_registry_lists_profiles():
    assert [p.name for p in profiles.PROFILES] == ["cic_ids2017", "nsl_kdd", "generic"]
    assert all(p.title for p in profiles.PROFILES)


def test_detection_ignores_header_case_and_spacing(tmp_path):
    header = "DURATION, Protocol_Type ,service,flag,src_bytes,dst_bytes,attack"
    path = tmp_path / "k.csv"
    path.write_text(header + "\n0,tcp,http,SF,1,2,normal\n", encoding="utf-8")
    match = inspect_dataset(path).profile
    assert match.profile.name == "nsl_kdd"
    assert match.variant == "column_subset"
