# Test fixtures

Small files used by the test suite. Rebuild them with:

```
python tests/fixtures/make_fixtures.py
```

The real-data samples are cut from the local datasets in `data/raw/`, which are never committed. The synthetic files use a fixed seed, so a rebuild produces the same files.

| File | Source | Purpose |
|---|---|---|
| `cic_ddos_sample.csv` | `data/raw/DDOS(CIC-2017)_FILTERED_TEST.csv`, 100 BENIGN + 100 DDoS rows, evenly spaced | Loader and profile tests. The header is kept **exactly** as in the source, including the leading spaces (B17). This is a *filtered* variant of the original CIC-IDS2017 CSVs (15 columns, no `Timestamp`/IP/`Flow ID`) |
| `cic2017_original_sample.csv` | `data/raw/CIC-2017.csv` (original CIC-IDS2017 "MachineLearningCSV", 79 columns): 8 rows per label for all 15 labels, plus rows containing `inf` and empty cells | Loader/profile tests against the **real** full format: leading-space header, `Fwd Header Length.1`, `Web Attack � …` labels, `inf`/empty cells, no time/IP columns. See `docs/datasets.md` |
| `nslkdd_train_sample.txt` | `data/raw/KDDTrain+.txt`, up to 10 rows per attack label (23 labels) | NSL-KDD profile with a header, including the label-derived `level` column (L6) |
| `nslkdd_test_sample_noheader.txt` | `data/raw/KDDTest+.txt`, 100 rows, evenly spaced | NSL-KDD test file **without a header**, like the source (L10) |
| `label_leak.csv` | Synthetic (seed 42), 300 rows, 3 classes | Leakage-guard tests (D9): label copy / encoded / prefixed (block), strong continuous feature (warn), `row_id` (ID-like), `Idle Mean` (must **not** trip the `id` pattern) |
| `xss_header.csv` | Synthetic | XSS tests (S5): HTML/JS payloads in the header and cells |

## CIC-IDS2017 variants covered

| Variant | Fixture | Status |
|---|---|---|
| Original "MachineLearningCSV" (full, 79 columns) | `cic2017_original_sample.csv` | From the real file |
| Filtered column subset of the original (15 columns) | `cic_ddos_sample.csv` | From the real file |
| "Improved" re-labelled version (`Timestamp`, IPs, `Attempted Category`) | — (built inside `tests/ml/test_profiles.py`) | **Full header not available locally.** The test builds a partial header only from the 8 real feature names of the v4 model, which was trained on this version (recorded in `docs/baseline/cli_v4.json`; the model's `config.json` was deleted in step 0.5). A full fixture is added only from a real download, never reconstructed from memory, because a wrong header fixture would make the profile tests pass against a format that doesn't exist |
