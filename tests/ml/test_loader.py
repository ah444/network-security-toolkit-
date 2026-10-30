"""Step 0.4a: dataset loader (sniffed separator, stripped headers, duplicates, dtypes) [D16, B17, R7]."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from toolkit.core.services.ids import datasets
from toolkit.core.data.ids.dataset_files import DatasetError
from toolkit.core.services.ids.datasets import load_dataset


def _write(path, text, encoding="utf-8"):
    path.write_bytes(text.encode(encoding))
    return path


def test_headers_stripped_types_and_duplicates(tmp_path):
    text = (
        " Flow Duration; Fwd Header Length ;Flow Bytes/s;Exact Half;Label; Fwd Header Length.1\n"
        "100000;20;0.1;0.5;BENIGN;20\n"
        "5;-3;inf;1.25;DDoS;-3\n"
        "7;100;;2.0;BENIGN;100\n"
    )
    loaded = load_dataset(_write(tmp_path / "flows.csv", text))
    frame, report = loaded.frame, loaded.report

    assert list(frame.columns) == ["Flow Duration", "Fwd Header Length", "Flow Bytes/s",
                                   "Exact Half", "Label"]
    assert report.separator == ";"
    assert report.renamed_columns[" Flow Duration"] == "Flow Duration"
    assert report.renamed_columns[" Fwd Header Length "] == "Fwd Header Length"
    assert report.dropped_duplicate_columns == ("Fwd Header Length.1",)

    assert frame["Flow Duration"].dtype == np.int32
    assert frame["Fwd Header Length"].dtype == np.int8
    assert frame["Flow Bytes/s"].dtype == np.float64, "0.1 isn't exact in float32: keep float64"
    assert frame["Exact Half"].dtype == np.float32, "lossless float32 downcast"
    assert isinstance(frame["Label"].dtype, pd.CategoricalDtype)
    assert report.downcast["Flow Duration"] == "int32"
    assert "Flow Bytes/s" not in report.downcast

    assert report.rows == 3
    assert report.missing_values == 1
    assert report.infinite_values == 1


def test_suffixed_column_kept_when_values_differ(tmp_path):
    text = "a,b,a.1,Label\n1,2,3,x\n4,5,6,y\n"
    loaded = load_dataset(_write(tmp_path / "t.csv", text))
    assert "a.1" in loaded.frame.columns
    assert loaded.report.dropped_duplicate_columns == ()
    assert "SUFFIXED_COLUMN_DIFFERS" in loaded.report.warnings


def test_names_made_unique_after_stripping(tmp_path):
    loaded = load_dataset(_write(tmp_path / "t.csv", " a,a,Label\n1,2,x\n3,4,y\n"))
    assert list(loaded.frame.columns) == ["a", "a.1", "Label"]
    assert "SUFFIXED_COLUMN_DIFFERS" in loaded.report.warnings


@pytest.mark.parametrize("separator", [",", ";", "\t", "|"])
def test_separator_sniffed(tmp_path, separator):
    rows = [["f1", "f2", "Label"], ["1", "2.5", "BENIGN"], ["3", "4.5", "DDoS"]]
    text = "\n".join(separator.join(r) for r in rows) + "\n"
    loaded = load_dataset(_write(tmp_path / "t.txt", text))
    assert loaded.report.separator == separator
    assert list(loaded.frame.columns) == ["f1", "f2", "Label"]
    assert loaded.frame["f2"].tolist() == [2.5, 4.5]


def test_quoted_field_containing_separator(tmp_path):
    text = 'name,value,Label\n"a, b",1,x\n"c",2,y\n'
    loaded = load_dataset(_write(tmp_path / "t.csv", text))
    assert loaded.report.separator == ","
    assert loaded.frame["name"].tolist() == ["a, b", "c"]


def test_chunked_read_equals_single_read(fixtures_dir):
    path = fixtures_dir / "cic2017_original_sample.csv"
    whole = load_dataset(path)
    chunked = load_dataset(path, chunk_rows=7)
    pd.testing.assert_frame_equal(whole.frame, chunked.frame)
    assert whole.report == chunked.report


def test_sha256_matches_file_bytes(fixtures_dir):
    path = fixtures_dir / "cic_ddos_sample.csv"
    assert load_dataset(path).report.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "cp1252"])
def test_encodings(tmp_path, encoding):
    text = "Flow Duration,Label\n1,BENIGN\n2,Web Attack – XSS\n"
    loaded = load_dataset(_write(tmp_path / "t.csv", text, encoding))
    assert list(loaded.frame.columns) == ["Flow Duration", "Label"], "BOM must not stick to a name"
    assert loaded.frame["Label"].tolist()[1] in {"Web Attack – XSS", "Web Attack - XSS"}


def test_real_cic_sample(fixtures_dir):
    loaded = load_dataset(fixtures_dir / "cic2017_original_sample.csv")
    frame, report = loaded.frame, loaded.report
    assert frame.shape == (126, 78)
    assert report.dropped_duplicate_columns == ("Fwd Header Length.1",)
    assert not any(name != name.strip() for name in frame.columns)
    labels = set(frame["Label"].astype(str))
    assert "Web Attack - Brute Force" in labels
    assert not any("�" in label for label in labels)
    assert report.label_values_normalised > 0
    assert report.infinite_values > 0 and report.missing_values > 0
    port = frame["Destination Port"].dtype
    assert port.kind == "i" and port.itemsize <= 4


def test_nslkdd_headerless_loaded_with_names(fixtures_dir):
    with_header = load_dataset(fixtures_dir / "nslkdd_train_sample.txt")
    headerless = load_dataset(fixtures_dir / "nslkdd_test_sample_noheader.txt")
    assert headerless.report.has_header is False
    assert list(headerless.frame.columns) == list(with_header.frame.columns)
    assert len(headerless.frame) == 100, "first data row must not be eaten as a header"
    assert headerless.frame.columns[-2:].tolist() == ["attack", "level"]
    assert isinstance(headerless.frame["protocol_type"].dtype, pd.CategoricalDtype)


def test_empty_file_raises(tmp_path):
    with pytest.raises(DatasetError, match="empty"):
        load_dataset(_write(tmp_path / "e.csv", ""))


def test_header_only_raises(tmp_path):
    with pytest.raises(DatasetError, match="no data rows"):
        load_dataset(_write(tmp_path / "h.csv", "a,b,Label\n"))


def test_ragged_rows_raise_dataset_error(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(_write(tmp_path / "r.csv", "a,b,Label\n1,2,x\n1,2,3,4,5\n"))


def test_inspect_reads_only_the_start(fixtures_dir):
    """inspect_dataset gives format + profile without loading the whole file."""
    info = datasets.inspect_dataset(fixtures_dir / "cic_ddos_sample.csv")
    assert info.separator == ","
    assert info.profile.profile.name == "cic_ids2017"
