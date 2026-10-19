"""Pin the shape of the test fixtures so later tests can rely on them."""

import csv
from collections import Counter


def _read(path, header=True):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    return (rows[0], rows[1:]) if header else (None, rows)


def test_cic_sample_keeps_original_header_and_classes(fixtures_dir):
    header, rows = _read(fixtures_dir / "cic_ddos_sample.csv")
    assert len(rows) == 200
    assert header[-1] == " Label"  # leading space kept on purpose (B17)
    assert any(name.startswith(" ") for name in header)
    assert Counter(row[-1].strip() for row in rows) == {"BENIGN": 100, "DDoS": 100}


def test_cic2017_original_sample_matches_real_format(fixtures_dir):
    header, rows = _read(fixtures_dir / "cic2017_original_sample.csv")
    assert len(header) == 79
    assert header[0] == " Destination Port" and header[-1] == " Label"
    assert " Fwd Header Length.1" in header  # duplicate column, as in the real file
    assert not {"Timestamp", " Timestamp", "Flow ID", " Flow ID"} & set(header)
    labels = {row[-1] for row in rows}
    assert len(labels) == 15
    assert "Web Attack � XSS" in labels  # replacement character kept on purpose
    assert any(cell.strip() == "inf" for row in rows for cell in row)
    assert any(cell.strip() == "" for row in rows for cell in row[:-1])


def test_nslkdd_train_sample_has_header_and_level_column(fixtures_dir):
    header, rows = _read(fixtures_dir / "nslkdd_train_sample.txt")
    assert header[-2:] == ["attack", "level"]
    assert len({row[-2] for row in rows}) >= 20  # many attack labels, incl. rare ones


def test_nslkdd_test_sample_has_no_header(fixtures_dir):
    _, rows = _read(fixtures_dir / "nslkdd_test_sample_noheader.txt", header=False)
    assert len(rows) == 100
    assert rows[0][0].isdigit()  # first line is data ("duration"), not a column name


def test_label_leak_fixture_columns(fixtures_dir):
    header, rows = _read(fixtures_dir / "label_leak.csv")
    assert header == [
        "row_id", "f_noise", "f_strong", "Idle Mean", "xyz_col", "enc_col", "pre_col", "Label"
    ]
    assert len(rows) == 300
    label = header.index("Label")
    assert all(row[header.index("xyz_col")] == row[label] for row in rows)
    assert all(row[header.index("pre_col")] == f"cls_{row[label]}" for row in rows)


def test_xss_fixture_contains_payloads(fixtures_dir):
    header, rows = _read(fixtures_dir / "xss_header.csv")
    assert any("<" in name for name in header)
    assert any("<script>" in cell for row in rows for cell in row)
