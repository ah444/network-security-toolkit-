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
- **Dataset decision (beta exit):** CIC-IDS2017 *is* available, so the NSL-KDD fallback is not needed for availability. Whether the "optimistic random split" caveat is acceptable for the CLI default, or whether the improved version should be downloaded for a temporal split, stays open until the beta-exit decision.

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

## Licences and citations (0.5b, still to complete)

| Dataset | Official page | Citation | Licence / redistribution |
|---|---|---|---|
| CIC-IDS2017 | https://www.unb.ca/cic/datasets/ids-2017.html | I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", ICISSP 2018 | **To confirm** from the official page before shipping a model trained on it |
| NSL-KDD | https://www.unb.ca/cic/datasets/nsl.html | M. Tavallaee, E. Bagheri, W. Lu, A. A. Ghorbani, "A Detailed Analysis of the KDD CUP 99 Data Set", IEEE CISDA 2009 | **To confirm** from the official page |

Raw datasets are never committed or redistributed by this project.
