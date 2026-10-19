"""Rebuild the small test fixtures in this folder.

Real-data samples are cut from the local datasets in ``data/raw/`` (not
committed); synthetic files are generated with a fixed seed, so running this
script again produces identical files.

Usage (from the project root):  python tests/fixtures/make_fixtures.py
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE.parent.parent / "data" / "raw"
SEED = 42


def _evenly_spaced(rows: list, count: int) -> list:
    """Pick ``count`` rows spread evenly over ``rows`` (deterministic)."""
    if len(rows) <= count:
        return rows
    step = len(rows) / count
    return [rows[int(i * step)] for i in range(count)]


def _write_csv(path: Path, header: list | None, rows: list) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if header is not None:
            writer.writerow(header)
        writer.writerows(rows)


def cic_ddos_sample() -> None:
    """200 rows (100 BENIGN + 100 DDoS) from the local CIC-IDS2017 DDoS file.

    The header is kept exactly as in the source, including the leading spaces.
    """
    source = RAW / "DDOS(CIC-2017)_FILTERED_TEST.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        by_label = defaultdict(list)
        for row in reader:
            by_label[row[-1].strip()].append(row)
    rows = _evenly_spaced(by_label["BENIGN"], 100) + _evenly_spaced(by_label["DDoS"], 100)
    _write_csv(HERE / "cic_ddos_sample.csv", header, rows)


def nslkdd_samples() -> None:
    """NSL-KDD train sample (with header) and test sample (no header, like the source)."""
    with (RAW / "KDDTrain+.txt").open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        by_label = defaultdict(list)
        for row in reader:
            by_label[row[-2]].append(row)
    # Up to 10 rows per attack label, so rare classes are represented too.
    rows = [row for label in sorted(by_label) for row in _evenly_spaced(by_label[label], 10)]
    _write_csv(HERE / "nslkdd_train_sample.txt", header, rows)

    with (RAW / "KDDTest+.txt").open(encoding="utf-8", newline="") as handle:
        test_rows = list(csv.reader(handle))
    _write_csv(HERE / "nslkdd_test_sample_noheader.txt", None, _evenly_spaced(test_rows, 100))


def label_leak() -> None:
    """Synthetic data for the leakage guard (plan D9).

    - ``xyz_col``  : exact copy of the label under an unfamiliar name  -> must be blocked
    - ``enc_col``  : label encoded as integers                         -> must be blocked
    - ``pre_col``  : label with a prefix                               -> must be blocked
    - ``f_strong`` : continuous, near-perfect but legitimate feature   -> warn only
    - ``row_id``   : integer row counter (ID-like)                     -> flagged as ID
    - ``Idle Mean``: ordinary feature containing "Id" in its name      -> NOT flagged
    - ``f_noise``  : uninformative feature                             -> OK
    """
    rng = random.Random(SEED)
    labels = ["BENIGN", "DDoS", "PortScan"]
    centres = {"BENIGN": 10.0, "DDoS": 50.0, "PortScan": 90.0}
    header = ["row_id", "f_noise", "f_strong", "Idle Mean", "xyz_col", "enc_col", "pre_col",
              "Label"]
    rows = []
    for i in range(300):
        label = labels[i % 3]
        rows.append([
            i,
            round(rng.gauss(0, 1), 6),
            round(rng.gauss(centres[label], 4.0), 6),
            round(rng.uniform(0, 1000), 3),
            label,
            labels.index(label),
            f"cls_{label}",
            label,
        ])
    _write_csv(HERE / "label_leak.csv", header, rows)


def xss_header() -> None:
    """CSV whose header and cells contain HTML/JS payloads (plan S5)."""
    header = ["<img src=x onerror=alert(1)>", '"><script>alert(2)</script>', "pkt_len",
              "Label"]
    rows = [
        ["<b>bold</b>", "<svg onload=alert(3)>", 60, "<script>alert(4)</script>"],
        ["plain", "text", 1500, "BENIGN"],
        ["x", "y", 40, "DDoS"],
    ]
    _write_csv(HERE / "xss_header.csv", header, rows)


if __name__ == "__main__":
    cic_ddos_sample()
    nslkdd_samples()
    label_leak()
    xss_header()
    print("fixtures written to", HERE)
