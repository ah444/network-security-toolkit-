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
| `nvd_openssh_8.9.json` | Real reply of the NVD CVE API 2.0 for `keywordSearch=OpenSSH 8.9`, saved 2026-10-20 | NVD parser tests (B10): CVSS v3.1 primary and secondary scores, CVSS v2 and SSVC entries |
| `nvd_no_results.json` | Real NVD reply with no results (`keywordSearch=OpenSSH 8.9p1`), saved 2026-10-20 | Empty-result handling |
| `label_leak.csv` | Synthetic (seed 42), 300 rows, 3 classes | Leakage-guard tests (D9): label copy / encoded / prefixed (block), strong continuous feature (warn), `row_id` (ID-like), `Idle Mean` (must **not** trip the `id` pattern) |
| `xss_header.csv` | Synthetic | XSS tests (S5): HTML/JS payloads in the header and cells |
| `legacy/old_names.bundle`, `legacy/old_names.expected.json` | Synthetic (seed 0), made by `legacy/make_legacy_bundle.py` with the code **before** the layered layout (v5.1 step 1.5d), signed with the test key `bytes(range(32))`; decision tree + logistic regression with signed log, scaling and the `f_classif` selector | `tests/ml/test_legacy_bundle.py`: a bundle whose pickle uses the old module names (`toolkit.ids_core.pipeline`, `toolkit.ids_core.registry`) still loads and predicts the same labels. `make_fixtures.py` doesn't rebuild it: newer code would write the new names (the script then stops) |
| `golden/outputs.json` | Written once by `golden/make_goldens.py` (v5.2 step 2.3a) from the code **before** the step 2.3 refactor; the script refuses to replace it | `tests/test_goldens.py`: the loader's report, column types and frame hash, the splits, the leakage report and the balancing recommendation for the fixtures above, and `balancing._rule` for all 288 combinations of its inputs, exactly as the old code gave them (`tests/golden_outputs.py` turns them into JSON) |
| `legacy/v51_names.bundle`, `legacy/v51_names.expected.json` | Synthetic (seed 0), made by `legacy/make_v51_bundle.py` with the v5.1 layout (v5.1 step 1.5d to v5.2 step 2.3a), before the code was arranged by interface; signed with the same test key, the same models | The same tests for the second generation of names (`toolkit.services.ids.pipeline`, `toolkit.services.ids.registry`), which every bundle saved in that time carries. Never rebuilt by newer code either |

## Citations

The real-data samples are redistributed under the datasets' own terms, which require these citations (details in `docs/datasets.md`, "Licences and citations"):

- **CIC-IDS2017:** I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", ICISSP 2018. https://www.unb.ca/cic/datasets/ids-2017.html
- **NSL-KDD:** the NSL-KDD dataset, and M. Tavallaee, E. Bagheri, W. Lu, A. A. Ghorbani, "A Detailed Analysis of the KDD CUP 99 Data Set", IEEE CISDA 2009. https://www.unb.ca/cic/datasets/nsl.html

## CIC-IDS2017 variants covered

| Variant | Fixture | Status |
|---|---|---|
| Original "MachineLearningCSV" (full, 79 columns) | `cic2017_original_sample.csv` | From the real file |
| Filtered column subset of the original (15 columns) | `cic_ddos_sample.csv` | From the real file |
| "Improved" re-labelled version (`Timestamp`, IPs, `Attempted Category`) | — (built inside `tests/ml/test_profiles.py`) | **Full header not available locally.** The test builds a partial header only from the 8 real feature names of the v4 model, which was trained on this version (recorded in `docs/baseline/cli_v4.json`; the model's `config.json` was deleted in step 0.5). A full fixture is added only from a real download, never reconstructed from memory, because a wrong header fixture would make the profile tests pass against a format that doesn't exist |
