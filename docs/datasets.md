# Datasets

All datasets live in `data/raw/` (gitignored, never committed). This file records **what each file is**, its **fingerprint** (SHA-256), what was measured in it, and what that means for training. Every TrainingRun (plan D21) refers to a dataset by the SHA-256 listed here.

Profiled on 2026-10-19 with a read-only streaming pass over each full file.

---

## 1. CIC-IDS2017, original "MachineLearningCSV" variant — `CIC-2017.csv`

| Property | Value |
|---|---|
| File | `data/raw/CIC-2017.csv` |
| Size | 963,373,754 bytes (≈ 919 MB) |
| SHA-256 | `642b251fd41f120cd1d8f9e25e348ed6d272ba468f061f03aeba1c9d4ce7e1ef` |
| Rows / columns | 2,830,743 rows × 79 columns (78 features + `Label`) |
| Variant | **Original** CIC-IDS2017 "MachineLearningCSV" files, all 8 capture days combined into one file. **Not** the "improved" re-labelled version |
| Header quirks | 65 of 79 names start with a space (e.g. `" Flow Duration"`, `" Label"`); duplicate column literally named `Fwd Header Length.1` |
| Time / IP / Flow ID columns | **None.** No `Timestamp`, `Src IP`, `Dst IP` or `Flow ID` |
| Encoding | Valid UTF-8, but the three web-attack labels contain the replacement character `�` (U+FFFD) where the original had an en-dash, e.g. `Web Attack � Brute Force` |

**Labels (15 classes):**

| Label | Rows | Unique rows | Duplicates removed |
|---|---:|---:|---:|
| BENIGN | 2,273,097 | 2,096,484 | 7.8 % |
| DoS Hulk | 231,073 | 172,849 | 25.2 % |
| PortScan | 158,930 | 90,819 | **42.9 %** |
| DDoS | 128,027 | 128,016 | 0.0 % |
| DoS GoldenEye | 10,293 | 10,286 | 0.1 % |
| FTP-Patator | 7,938 | 5,933 | 25.3 % |
| SSH-Patator | 5,897 | 3,219 | **45.4 %** |
| DoS slowloris | 5,796 | 5,385 | 7.1 % |
| DoS Slowhttptest | 5,499 | 5,228 | 4.9 % |
| Bot | 1,966 | 1,953 | 0.7 % |
| Web Attack � Brute Force | 1,507 | 1,470 | 2.5 % |
| Web Attack � XSS | 652 | 652 | 0 % |
| Infiltration | 36 | 36 | 0 % |
| Web Attack � Sql Injection | 21 | 21 | 0 % |
| Heartbleed | 11 | 11 | 0 % |
| **Total** | **2,830,743** | **2,522,362** | **10.9 %** (308,381 rows) |

**Data quality findings:**
- **308,381 exact duplicate rows (10.9 %).** With today's random split, many of them land in both train and test, which inflates the scores (L4). The duplicates are concentrated in PortScan, SSH-Patator, FTP-Patator and DoS Hulk.
- **1,358 empty cells and 4,376 infinite values**, written as `inf` in this file (in `Flow Bytes/s` and `Flow Packets/s`, from zero-duration flows).
- **Negative values** where they make no physical sense, e.g. `Flow Duration` (115 rows), `Flow IAT Min` (2,891), `Fwd Header Length` (35). `Init_Win_bytes_forward/backward` use `-1` as a "not set" marker (1.0 M / 1.4 M rows): a sentinel, not an error.
- **Extremely small classes:** Heartbleed 11, Sql Injection 21, Infiltration 36 rows. Any test split leaves only a handful of them, so their per-class scores are statistically meaningless. Options for training (decided in 0.4b/0.4d, recorded in the TrainingRun): keep and report with a warning, merge the three web attacks into one `Web Attack` class, or drop classes below a minimum count.

**What this means for the plan:**
- **Profile:** CIC "original" variant. Strip header spaces, drop `Fwd Header Length.1` as a duplicate, normalise `Web Attack � …` → `Web Attack - …`, and treat `inf` and empty cells explicitly.
- **Split strategy:** there is no time column, so `temporal_per_class` is impossible. Grouping by capture file is also impossible, because the 8 days were combined into one file with no file or day column. Per plan D9 this means **dedupe → stratified split, with a visible warning** that scores may be optimistic. The warning is written into the TrainingRun and the model card.
- **CLI model (0.5):** a CIC model trained on this file has **15 classes**, not the 16 of the old v4 model. The v4 classes (`… - Attempted`) come from the *improved* version, which is not available. The leaky `Attempted Category` feature doesn't exist in this file at all.
- **Dataset decision (beta exit, 2026-10-20): this file is the training dataset for the CLI model (0.5).** See "Dataset decision" below (revised in 0.5: no model is shipped).

---

## 2. CIC-IDS2017, filtered DDoS subset — `DDOS(CIC-2017)_FILTERED_TEST.csv`

| Property | Value |
|---|---|
| SHA-256 | `42381db8eaa54a36eeacda06ff3394d973d30cabe9bff7429fff2d3c165095ef` |
| Rows / columns | 225,745 rows × 15 columns (14 features + `Label`) |
| Variant | Column subset of the original CIC-IDS2017 Friday DDoS capture (same naming, leading spaces) |
| Labels | BENIGN, DDoS (binary) |
| Duplicates | 23,416 exact duplicate rows (225,745 → 202,329 unique) |
| Time / IP columns | None |
| Origin | Came with the project (was in `uploads/`); exact filtering steps unknown |

## 3. NSL-KDD — `KDDTrain+.txt`, `KDDTest+.txt`

| Property | `KDDTrain+.txt` | `KDDTest+.txt` |
|---|---|---|
| SHA-256 | `ff87bf1cc9dff67b23a15a3b7752d0aee2573cdc53210ed6517758e30202ebef` | `fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84` |
| Lines | 125,974 (header + 125,973 rows) | 22,544 rows, **no header** |
| Columns | 41 features + `attack` (label) + `level` (difficulty score) | same layout |
| Note | The header line was added to this copy; the official file has none | The official test set contains attack types not in the training set |

**Leakage note:** `level` is a difficulty score derived from how many classifiers got the record right. It must be **blocked** as a feature (L6). **Split:** official train/test files (plan D9 `official`).

---

## Profile detection (step 0.4a)

`toolkit/core/services/ids` (`toolkit/ids_core` before v5.1 step 1.5d) recognises each file from its content (header names, or value patterns for headerless files), never from the file name. Checked on 2026-10-20 with a full load of every file above:

| File | Profile / variant | Header | Label | Blocked as features | Split strategy | Warnings |
|---|---|---|---|---|---|---|
| `CIC-2017.csv` | `cic_ids2017` / `original` | yes | `Label` | — | `stratified` | `CIC_NO_TIME_COLUMN` |
| `DDOS(CIC-2017)_FILTERED_TEST.csv` | `cic_ids2017` / `original_subset` | yes | `Label` | — | `stratified` | `COLUMN_SUBSET`, `CIC_NO_TIME_COLUMN` |
| `KDDTrain+.txt` | `nsl_kdd` / `full` | yes | `attack` | `level` | `official` | — |
| `KDDTest+.txt` | `nsl_kdd` / `full` | **no**, names applied | `attack` | `level` | `official` | `HEADERLESS_NAMES_APPLIED` |

The loader's SHA-256, row counts, empty-cell and `inf` counts match the tables above. On `CIC-2017.csv` it also confirmed that `Fwd Header Length.1` is identical to `Fwd Header Length` in all 2,830,743 rows, so it is dropped, and it rewrote the 2,180 `Web Attack �` labels as `Web Attack -`.

**Not covered yet (no local file):** UNSW-NB15 and the full header of the "improved" CIC-IDS2017 version. Their schemas aren't reconstructed from memory. Until a real file is available, a UNSW CSV loads with the generic profile (`UNKNOWN_SCHEMA`), and the improved CIC version is recognised only from the 8 feature names of the v4 model (its `config.json` was deleted with the v4 pickles in step 0.5; the names are kept in `docs/baseline/cli_v4.json`).

---

## Dedupe and split (step 0.4b)

`toolkit/core/services/ids/split.py` with the profile's split strategy, checked on 2026-10-20 (default `SplitConfig`: 20 % test, seed 42):

| File | Copies removed | Conflicting rows¹ | Strategy | Train / test rows | Warnings |
|---|---|---|---|---|---|
| `CIC-2017.csv` | 308,381 (10.9 %) | 1,396 | `stratified` | 2,017,889 / 504,473 | `STRATIFIED_SCORES_MAY_BE_OPTIMISTIC` |
| `DDOS(CIC-2017)_FILTERED_TEST.csv` | 23,416 | 72 | `stratified` | 161,864 / 40,465 | `STRATIFIED_SCORES_MAY_BE_OPTIMISTIC` |
| `KDDTrain+.txt` + `KDDTest+.txt` | none (kept as published); 610 test rows also in train | 31 | `official` | 125,973 / 22,544 (strict mode: 125,973 / 21,934) | `OFFICIAL_TEST_OVERLAP_KEPT` (strict: `_REMOVED`), `CLASS_MISSING_FROM_TEST`, `CLASS_MISSING_FROM_TRAIN` |

¹ Same feature values, different labels: counted, not removed.

- **CIC:** every one of the 15 classes is on both sides, but the smallest are tiny: Heartbleed 9 / 2, Infiltration 29 / 7, Sql Injection 17 / 4. With no time column the split is stratified, so the scores may be optimistic.
- **NSL-KDD:** both files are used exactly as published by default, so results are comparable with other work. Neither file has copies inside it, but 610 KDDTest+ rows (2.7 %) also occur in KDDTrain+ (compared without `level`). They are reported, and removed only in strict mode (`SplitConfig(official_dedupe=True)`); default scores are therefore slightly optimistic. 17 attack types occur only in KDDTest+ (e.g. `apache2` 737, `mscan` 996), and `spy` (2) and `warezclient` (890) occur only in KDDTrain+. This is how the published files were designed (unseen attacks in test), so it is reported, not "fixed".

## Dataset decision (5.0-beta exit, 2026-10-20)

**Decision (user): the default CLI model of step 0.5 is trained on `CIC-2017.csv`, the original CIC-IDS2017 "MachineLearningCSV" variant** (SHA-256 in section 1). The NSL-KDD fallback (plan D12) is not used.

**Revised in step 0.5 (user decision, 2026-10-20): no model is shipped.** A full-file Random Forest export is about 2 GB, far over GitHub's 100 MB limit, and a size cap would limit which models can be used. The CLI now lists the models exported on this install and the user picks one; `CIC-2017.csv` stays the recommended training file.

Why:
- It is available locally, hashed, profile-detected (`cic_ids2017` / `original`) and split-validated (0.4b), so step 0.5 isn't blocked.
- It covers modern traffic and 15 classes. NSL-KDD comes from 1999-era traffic, and on its official test file macro-F1 drops to 0.26 because 17 attack types exist only there (0.4d real-data check).
- The 0.4d real-data check on this file: XGBoost selected by CV macro-F1 0.854 ± 0.043, test macro-F1 0.886, balanced accuracy 0.892.

Accepted caveats, written into the TrainingRun and the model card:
- **No time column**, so the split is dedupe → **stratified** (`STRATIFIED_SCORES_MAY_BE_OPTIMISTIC`): flows from the same attack session can be on both sides, so scores may be optimistic.
- **15 classes, not v4's 16**: v4's `… - Attempted` classes come from the improved version.
- **Tiny classes** can't be scored reliably: Heartbleed (9 train / 2 test after dedupe), Sql Injection (17 / 4), Infiltration (29 / 7).

Revisit: if the improved CIC-IDS2017 (with timestamps) is downloaded, its header is added to the CIC profile from the real file, and the model can be retrained with a `temporal_per_class` split in a later release.

## Licences and citations (0.5b)

Checked on 2026-10-20 on each dataset's official page. Words in quotation marks are the page's own. None of the pages names a standard licence (such as a Creative Commons one); the terms below are all they state. The project's own code licence (README) does not cover any dataset.

### CIC-IDS2017 (files 1 and 2)

| | |
|---|---|
| Official page | https://www.unb.ca/cic/datasets/ids-2017.html |
| Terms | The dataset page says the datasets are "publicly available for researchers". The FAQ on the CIC datasets index (https://www.unb.ca/cic/datasets/index.html) applies to all CIC datasets: "you may redistribute, republish, and mirror our datasets in any form", and "any use or redistribution of the data must include a citation to the dataset and the research paper listed" |
| Citation (required) | I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018 |
| Raw data | Redistribution allowed, with the citation |
| Derived models | Not mentioned on either page. Treated like redistributed data: allowed, with the citation |
| Filtered DDoS subset (file 2) | A column subset of CIC-IDS2017, so the same terms and citation apply. It came with the project and its exact origin is unknown (section 2) |

### NSL-KDD (file 3)

| | |
|---|---|
| Official page | https://www.unb.ca/cic/datasets/nsl.html (the page now says the dataset "is no longer available" there; the local copy was already in the project) |
| Terms | "You may redistribute, republish, and mirror the ISCX NSL-KDD dataset in any form." "However, any use or redistribution of the data must include a citation to the NSL-KDD dataset and the paper referenced below." |
| Citation (required) | The NSL-KDD dataset, and M. Tavallaee, E. Bagheri, W. Lu, A. A. Ghorbani, "A Detailed Analysis of the KDD CUP 99 Data Set", Second IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA), 2009 |
| Raw data | Redistribution allowed, with the citation |
| Derived models | Not mentioned. Treated like redistributed data: allowed, with the citation |

### UNSW-NB15 (no local file)

| | |
|---|---|
| Official page | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| Terms | "Free use of the UNSW-NB15 dataset for academic research purposes is hereby granted in perpetuity." "Use for commercial purposes should be agreed by the authors." The authors "have asserted their rights under the Copyright" |
| Citation (required) | The page asks users to cite five papers: (1) N. Moustafa, J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)", MilCIS, IEEE, 2015. (2) N. Moustafa, J. Slay, "The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 dataset and the comparison with the KDD99 dataset", Information Security Journal: A Global Perspective, 2016. (3) N. Moustafa et al., "Novel geometric area analysis technique for anomaly detection using trapezoidal area estimation on large-scale networks", IEEE Transactions on Big Data, 2017. (4) N. Moustafa et al., "Big data analytics for intrusion detection system: statistical decision-making using finite dirichlet mixture models", Data Analytics and Decision Support for Cybersecurity, Springer, 2017, pp. 127-156. (5) M. Sarhan, S. Layeghy, N. Moustafa, M. Portmann, "NetFlow Datasets for Machine Learning-Based Network Intrusion Detection Systems", BDTA 2020 / WiCON 2020, Springer, p. 117. The page lists papers 3 and 4 with "et al." only |
| Raw data | The page grants use, not redistribution: **do not redistribute** without the authors' agreement |
| Derived models | Academic research use only; anything commercial needs the authors' agreement |
| In this project | No file, no fixture, no model. Only the profile's detection rule exists, and its header waits for a real download (section "Profile detection") |

### Kaggle and other mirrors

A mirror (for example on Kaggle) may show its own licence label, but it can't grant more than the original authors did. Any copy of these datasets follows the original terms above, and a TrainingRun records the file's SHA-256, not the mirror.

### What this project redistributes

| What | Status |
|---|---|
| Raw dataset files | Never committed (`data/raw/` is gitignored) |
| Test fixtures | Small real samples of CIC-IDS2017 (2 files) and NSL-KDD (2 files) are committed in `tests/fixtures/`. Both datasets allow redistribution with a citation, so `tests/fixtures/README.md` carries both citations. No UNSW-NB15 rows are committed |
| Models | None shipped (0.5). Exports stay in `instance/bundles` on the install that made them. If a model is ever shipped, it carries its dataset's citation, and a UNSW-NB15 model is for academic use only |
