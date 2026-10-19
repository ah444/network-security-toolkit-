# Test fixtures

Small files used by the test suite. Rebuild them with:

```
python tests/fixtures/make_fixtures.py
```

The real-data samples are cut from the local datasets in `data/raw/`, which are never committed. The synthetic files use a fixed seed, so a rebuild produces the same files.

| File | Source | Purpose |
|---|---|---|
| `cic_ddos_sample.csv` | `data/raw/DDOS(CIC-2017)_FILTERED_TEST.csv`, 100 BENIGN + 100 DDoS rows, evenly spaced | Loader and profile tests. The header is kept **exactly** as in the source, including the leading spaces (B17). This is a *filtered* variant of the original CIC-IDS2017 CSVs (15 columns, no `Timestamp`/IP/`Flow ID`) |
| `nslkdd_train_sample.txt` | `data/raw/KDDTrain+.txt`, up to 10 rows per attack label (23 labels) | NSL-KDD profile with a header, including the label-derived `level` column (L6) |
| `nslkdd_test_sample_noheader.txt` | `data/raw/KDDTest+.txt`, 100 rows, evenly spaced | NSL-KDD test file **without a header**, like the source (L10) |
| `label_leak.csv` | Synthetic (seed 42), 300 rows, 3 classes | Leakage-guard tests (D9): label copy / encoded / prefixed (block), strong continuous feature (warn), `row_id` (ID-like), `Idle Mean` (must **not** trip the `id` pattern) |
| `xss_header.csv` | Synthetic | XSS tests (S5): HTML/JS payloads in the header and cells |

## Still to add (needs the real CIC-IDS2017 download)

Plan 0.4a needs header fixtures for **at least two published CIC-IDS2017 variants**: the original "MachineLearningCSV" files and the "improved" version with `Timestamp`/`Attempted Category`. They are added **from the real files** once the dataset is downloaded. They are not reconstructed from memory, because a wrong header fixture would make the profile tests pass against a format that doesn't exist.
