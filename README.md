# 🛡️ Network Security Toolkit

A comprehensive network security tool combining **Network Reconnaissance** and **Machine Learning-based Intrusion Detection System (IDS)**.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.1-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.14-red)

---

## ⚠️ Project status — read first

This project is being reworked into **v5.0 ("correct & safe")**; see [docs/IMPROVEMENT_PLAN.md](docs/IMPROVEMENT_PLAN.md) and the day-to-day [docs/V5.0_CHECKLIST.md](docs/V5.0_CHECKLIST.md). A measured baseline of the current version ([docs/baseline/FINDINGS.md](docs/baseline/FINDINGS.md)) showed that **earlier accuracy claims were not valid**:

- **IDS CLI model (v4):** it uses `Attempted Category`, a feature derived from the attack label itself. Changing only that value flips the prediction (e.g. `-1` turns every sample into BENIGN), and none of the CLI's 4 built-in sample flows is classified as intended. **The previously stated ~97–98 % accuracy is withdrawn.** Its files were deleted in v5.0 step 0.5: the CLI now uses the models you train and export in the fixed web trainer, and shows each one's real metrics from its training record (see the CLI's [model card](<tools/IDS/IDS_CLI - CIC-2017/models/README.md>)).
- **Web IDS trainer (v4):** its results were **optimistic**. "Apply SMOTE" and "Recommend Features" ran on the *whole* dataset before the train/test split, duplicate rows could land in both train and test, and label-derived columns (e.g. NSL-KDD `level`) were not blocked. The recommender even ranked `level` among the top 10 features. The app reported accuracy and weighted F1 only, which hide weak minority classes: on NSL-KDD, random forest showed 99.8 % accuracy but 76.6 % macro-F1. **Since 5.0-rc1 (step 0.4g) the web trainer runs on the leakage-free ML core** described under "ML core" below. The same datasets and scenarios re-run on that core, before vs after, are in the **ML report**: [docs/report/FINDINGS.md](docs/report/FINDINGS.md).

### Known limitations (current version)
- Web trainer scores on CIC-IDS2017 come from a stratified (random) split, because the file has no time column: flows of the same attack can be on both sides, so the scores may still be optimistic (the app says so). NSL-KDD uses its official test file.
- Exported models are signed **`.bundle`** files that load only on the toolkit install that made them. The signing key is created in `instance/signing.key` on the first export: back up the `instance/` folder. If the key is lost, the bundles can't be loaded any more; retrain from the run's record in `instance/runs/`. Loading bundles from another computer comes in v5.1.
- Training runs inside the request: the page waits without a progress bar or cancel button (background jobs come in v5.1). On the full CIC file this takes many minutes and about 6 GB of memory.
- Web trainer data lives in memory, one workspace per browser session (tabs of the same browser share it). It is lost when the app restarts, removed after 240 minutes without use (`TOOLKIT_WORKSPACE_IDLE_MIN`), or with the **Clear workspace** button. Only one training runs at a time per workspace. If you serve the app over HTTPS, set `TOOLKIT_HTTPS=1` so the session cookie is marked `Secure`.
- Port scanner, what leaves your computer: CVE lookups send the detected software name and version (for example `OpenSSH 8.9`) to the public NVD API. The optional AI explanation (OpenRouter) and the AbuseIPDB check run only after you agree in that scan, and the AI gets the same software name only, never the raw banner.
- CVEs come from an NVD **keyword search** on that name and version, so the list can include CVEs that don't affect the exact build, or miss some. Severity comes from the CVSS scores only. Lookups run one at a time with a 15 s timeout each; without an NVD key, NVD allows about one request every 6 seconds, so scans with many open ports are slow (parallel lookups come in v5.1).
- The OS guess comes from the ping TTL, so it is only a rough family guess (Windows, Linux/macOS/BSD, network device). Hosts that don't answer ping, and IPv6 targets on Windows (no TTL shown), get no guess. On Windows, closed and firewalled TCP ports look the same, so both are reported as closed. A UDP port that doesn't answer is reported as `open|filtered`.
- Scans are limited to 256 hosts and 65,536 host×port checks (`TOOLKIT_SCAN_MAX_HOSTS`, `TOOLKIT_SCAN_MAX_PROBES`), and each one asks you to confirm you are authorised (`--i-am-authorized` in scripts).
- No model is shipped with the project: train and export one in the web trainer first. The CLI lists the exports in `instance/bundles` and opens only bundles exported on this install. Large models (for example Random Forest on the full CIC file, about 2 GB) take minutes to load and need several times their size in memory. A bundle made with a different scikit-learn, imbalanced-learn, XGBoost or LightGBM minor version is refused (retrain it); smaller version differences are shown as a warning.
- Starting the web trainer from the launcher menu prints its address (`http://127.0.0.1:5000/`) but doesn't open a browser; stop it with Ctrl+C to get back to the menu.
- Tested on Windows with Python 3.14 (419 automated tests, a fresh install followed by a scripted walk-through of every menu path, `scripts/menu_walkthrough.py`, and a browser session through the whole web trainer on real CIC-IDS2017 and NSL-KDD data, see `docs/release/`). **Linux and macOS are not tested on real machines:** the OS-specific logic (ping options, paths, file encodings) is unit-tested only.
- Only scan systems you are authorised to scan.

---

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Web IDS Features](#-web-ids-features-detailed)
- [Project Structure](#-project-structure)
- [Machine Learning Models](#-machine-learning-models)
- [API Keys Setup](#-api-keys-setup)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Features

### 1. Network Reconnaissance (Port Scanner)
- 🔍 TCP/UDP port scanning
- 🖥️ OS detection via TTL analysis
- 🏷️ Service banner grabbing
- 🌐 IPv4, IPv6, networks (CIDR) and domain names
- 🔐 CVE lookup via the NVD CVE API 2.0, with CVSS scores; severity comes from CVSS only
- 🤖 Optional AI explanation (OpenRouter), only after you agree; it receives the software name and version only, and its reply is shown as plain text marked "may be wrong"
- 📊 IP reputation check (via AbuseIPDB), only after you agree
- 📁 Export results to JSON/CSV/TXT (UTF-8, in `instance/scans/`, never overwritten, safe to open in a spreadsheet)

### 2. Intrusion Detection System (CLI)
- 🧠 Predicts with any model bundle exported by the web trainer (features, classes and best model come from the bundle)
- 📊 Uses the models you export from the web trainer: it lists them (newest first) and you pick one; each shows its real metrics from its training record
- ⌨️ Manual input (prefilled from a real example), real test-split examples, model switching
- 📄 `--csv` batch prediction (writes `<file>.predictions.csv`, never overwritten), `--json` input, `--info` model card

### 3. Intrusion Detection System (Web) ⭐ NEW FEATURES
- 🌐 Web-based UI with Flask
- 📤 Upload custom datasets (CSV/TXT)
- 📊 **Data Preview** - View first N rows with statistics
- 🎛️ **Dynamic model training** - Train on any dataset
- 📈 **Smart Data Analysis** - Automatic imbalance detection
- 🤖 **Intelligent Recommendations** - Best balancing method suggestion
- 🎚️ **8 Balancing Methods** - SMOTE, ADASYN, SMOTE-ENN, etc.
- 🎯 **Target Value Filtering** - Remove unwanted classes
- 📉 **Confusion Matrix** - Visual model evaluation
- 🔀 **Train/Test Split Slider** - Customize data splits
- 💾 **Export / Load Models** - signed `.bundle` files (all models of a run plus its training record)

---

## 🚀 Installation

### Prerequisites
- Python 3.11 or newer (developed and tested with Python 3.14 on Windows)
- pip

### Steps
Everything runs locally; no account or online service is needed for the IDS.

```bash
# 1. Open a terminal in the project folder (the folder that contains main.py)

# 2. Create a virtual environment in .venv
python -m venv .venv

# 3. Activate it
#    Windows (PowerShell):  .venv\Scripts\Activate.ps1
#    Windows (cmd):         .venv\Scripts\activate.bat
#    Linux / macOS:         source .venv/bin/activate

# 4. Install the pinned dependencies
python -m pip install -r requirements.txt
```

### Dependencies
Versions are **pinned** so results are reproducible and the bundled models load:
- `requirements.txt`: runtime libraries (Flask, waitress, pandas, numpy, scikit-learn, imbalanced-learn, XGBoost, LightGBM, requests)
- `requirements-dev.txt`: adds the development tools (pytest, ruff, pip-audit)
- `requirements.lock`: the complete environment, for an exact rebuild (`python -m pip install -r requirements.lock`)

### Datasets
Datasets are **not** part of the repository. Put them in `data/raw/`; see [docs/datasets.md](docs/datasets.md) for the expected files, their SHA-256 fingerprints and known data-quality issues.

### Development & tests
```bash
python -m pip install -r requirements-dev.txt
python -m pytest              # run the test suite
python -m ruff check .        # lint
python -m pip_audit -r requirements.lock   # check dependencies for known vulnerabilities
git config core.hooksPath .githooks         # once per clone: run lint + tests before every commit
```

---

## 📖 Usage

### Main Menu
```bash
python main.py
```

This launches an interactive menu with options:
1. **Network Reconnaissance** - Port scanning and vulnerability detection
2. **Intrusion Detection (CLI)** - Command-line ML-based IDS
3. **Intrusion Detection (Web)** - Web-based IDS trainer
4. **Exit**

### Network Reconnaissance
```bash
# Run directly
python tools/Port_scan/Port_Scan.py
```

Features:
- Enter target IP (IPv4 or IPv6), network (CIDR) or domain
- Specify port range (1-65535)
- Choose TCP/UDP/both protocols
- Confirm you are authorised to scan the target (or pass `--i-am-authorized`)
- View vulnerability reports
- Export results

### IDS CLI
```bash
# Interactive: lists your exports from the web trainer, you pick one
python "tools/IDS/IDS_CLI - CIC-2017/IDS_with_Machine_Learning.py"
# Label every row of a CSV file (with the newest export)
python "tools/IDS/IDS_CLI - CIC-2017/IDS_with_Machine_Learning.py" --csv flows.csv
# A specific export
python "tools/IDS/IDS_CLI - CIC-2017/IDS_with_Machine_Learning.py" --bundle "instance/bundles/<run id>.bundle"
```

Train and export a model in the web trainer first; `--list` shows your exports and `--info` the chosen model's card.

### IDS Web Interface
```bash
# Run Flask server
python tools/IDS/IDS_Web_CIC_Dynamic/app.py
```

Then open `http://localhost:5000` in your browser.

---

## 🌐 Web IDS Features (Detailed)

### 📊 Data Preview
After uploading a dataset, view:
- First N rows of data (configurable)
- Total rows & columns count
- Numeric vs Categorical columns
- Interactive table view

### 📈 Smart Data Analysis
The system analyzes your dataset; the recommendation below uses the **training split only**:

| Metric | Description |
|--------|-------------|
| **Imbalance Ratio** | Minority/Majority class ratio |
| **Severity** | mild (>50%), moderate (10-50%), severe (1-10%), extreme (<1%) |
| **Dataset Size** | small (<1K), medium (1K-10K), large (10K-100K), very_large (>100K) |
| **Dimensionality** | low (<20), medium (20-100), high (>100 features) |
| **Classes** | Binary (2) or Multi-class (>2) |

### 🤖 Intelligent Recommendations
Based on your data characteristics, the system recommends the best balancing method:

| Data Characteristics | Recommendation |
|---------------------|----------------|
| Balanced (>50% ratio) | None |
| Small dataset (<1000 samples) | Class Weights |
| Many classes (>10) + Extreme imbalance | Class Weights |
| Multi-class + Very Large + Severe | SMOTE-ENN |
| High dimensionality (>100 features) | Class Weights |
| Minority class < 6 samples | Class Weights (SMOTE needs ≥6) |

### 🎚️ 8 Balancing Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **None** | No balancing | Balanced datasets |
| **Class Weights** ⭐ | Adjusts model weights | Fast, safe choice |
| **SMOTE** | Synthetic oversampling | Medium datasets |
| **SMOTE-ENN** | SMOTE + noise removal | Large datasets |
| **ADASYN** | Adaptive synthetic sampling | Complex boundaries |
| **Borderline-SMOTE** | Focus on boundaries | Overlapping classes |
| **Random Undersampling** | Remove majority samples | Very large datasets |
| **Tomek Links** | Clean boundaries | Noisy data |
| **SMOTE + Tomek** | Combined approach | Balanced solution |

### 🎯 Target Value Filtering
Remove specific classes before training:
- View all class/label values with counts
- Select values to remove (e.g., "BENIGN" to focus on attacks)
- Dataset updates automatically

### 📉 Confusion Matrix
Visual evaluation of model performance:
- Select a trained model
- Shown for the test split (models are compared by cross-validation on the training rows; there is no separate validation split any more)
- View matrix with color-coded cells
- See True Positives, False Positives, etc.

### 🔀 Train/Test Split Slider
Customize data splits:
- Slider range: 10% to 50% for the test split (default 20% → Train: 80% | Test: 20%)
- Datasets with an official test file (NSL-KDD) use that file instead
- Real-time label update

---

## 📁 Project Structure

```
Project/
├── main.py                          # Main entry point
├── requirements.txt                 # Pinned runtime dependencies
├── requirements-dev.txt             # + pytest, ruff, pip-audit
├── requirements.lock                # Exact environment
├── pyproject.toml                   # pytest / ruff configuration
├── .gitignore                       # Git ignore rules
├── .env.example                     # Optional API keys template (copy to .env)
├── .githooks/pre-commit             # Lint + tests before every commit
├── README.md                        # This file
│
├── data/raw/                        # Local datasets (gitignored, see docs/datasets.md)
├── docs/                            # Improvement plan, v5.0 checklist, datasets, baseline, ML report
├── scripts/baseline_v4.py           # Reproduces the "before" baseline (docs/baseline/)
├── scripts/ml_report.py             # Re-runs it on the v5 ML core (docs/report/)
├── tests/                           # pytest suite + small fixtures
│
├── toolkit/                         # Shared package (no printing, no prompts)
│   ├── config.py                    # Settings, paths, .env reader
│   ├── logging_setup.py             # Log files in instance/logs, secrets redacted
│   ├── ports.py                     # ModelStore, IntelProvider, ProgressSink
│   ├── bundle_files.py, run_files.py  # Signed model bundles, training records
│   ├── ids_core/                    # Leakage-free ML core (see "ML core" below)
│   └── recon/                       # Scanner logic: targets, os_detect, probes registry,
│       └── intel/                   #   scanner, export; intel: NVD, OpenRouter, AbuseIPDB
│
└── tools/
    ├── Port_scan/                   # Network Reconnaissance
    │   ├── Port_Scan.py             # Port scanner (questions and printing; logic in toolkit/recon)
    │   └── api_keys.example.txt     # API keys template
    │
    └── IDS/
        ├── IDS_CLI - CIC-2017/      # CLI-based IDS
        │   ├── IDS_with_Machine_Learning.py
        │   └── models/
        │       └── README.md        # Where its models come from (your exports)
        │
        └── IDS_Web_CIC_Dynamic/     # Web-based IDS
            ├── app.py               # Flask application
            ├── templates/           # HTML templates
            │   ├── home.html
            │   └── dynamic_ids.html
            ├── static/
            │   ├── css/
            │   │   ├── style.css
            │   │   └── style2.css
            │   └── js/
            │       └── main.js
            ├── uploads/             # Uploaded datasets
            ├── dataset/             # Sample datasets
            └── (exports: instance/bundles/, run records: instance/runs/)
```

---

## 🤖 Machine Learning Models

### Datasets
- **Used and profiled:** CIC-IDS2017 (original "MachineLearningCSV" variant and a filtered DDoS subset) and NSL-KDD. Details, SHA-256 fingerprints and data-quality issues are in [docs/datasets.md](docs/datasets.md).
- **Not tested yet:** CIC-IDS2018, UNSW-NB15 and other CSV/TXT files. They can be uploaded to the web trainer, but they haven't been verified.

### Algorithms
| Algorithm | Type | Strengths |
|-----------|------|-----------|
| **Random Forest** | Ensemble | High accuracy, robust, handles imbalanced data |
| **Logistic Regression** | Linear | Fast, interpretable, good baseline |
| **K-Nearest Neighbors** | Instance-based | Simple, no training phase |
| **Support Vector Machine** | Kernel-based | Effective in high-dimensional spaces |
| **Decision Tree** | Tree-based | Interpretable, fast inference |
| **Naive Bayes** | Probabilistic | Fast, handles large datasets |
| **Gradient Boosting** | Ensemble | High accuracy, sequential learning |
| **AdaBoost** | Ensemble | Combines weak learners effectively |
| **Extra Trees** | Ensemble | Fast, randomized splits |
| **XGBoost** *(optional)* | Boosting | State-of-the-art performance |
| **LightGBM** *(optional)* | Boosting | Fast training, memory efficient |

> **Note:** XGBoost and LightGBM are included in `requirements.txt`.

### Attack Classes
- **CLI:** the classes of the export you choose (whatever the web trainer was trained on).
- **Web trainer:** whatever classes the uploaded dataset contains. The original CIC-IDS2017 file has 15 (BENIGN, DoS Hulk, PortScan, DDoS, DoS GoldenEye, FTP-Patator, SSH-Patator, DoS slowloris, DoS Slowhttptest, Bot, three Web Attack types, Infiltration, Heartbleed); see [docs/datasets.md](docs/datasets.md).

### Training Pipeline (web trainer, v5.0)
1. **Data Upload** - CSV/TXT; the dataset (CIC-IDS2017, NSL-KDD or generic) is recognised and its label preselected. An NSL-KDD train file uploaded together with a file named "*test*" uses the official split
2. **Preview** - View data statistics
3. **Class Analysis** - distribution of the whole dataset; the balancing recommendation comes from the training split only
4. **Filter Classes** - Remove unwanted targets (optional)
5. **Feature Recommendation** - the leakage guard first, then scores on the training split only
6. **Balancing** - 9 choices including "Recommended"; a preview shows the training split's class counts before and after; training applies the method inside each model's pipeline
7. **Split** - exact duplicates removed, then the dataset's split (stratified, or the official test file)
8. **Model Selection** - Choose from 11 algorithms
9. **Training** - 3-fold cross-validation on the training rows, models trained one after another
10. **Evaluation** - leaderboard with CV macro-F1 (mean ± std), test macro-F1, balanced accuracy and weakest-class recall; the test split is scored once; test confusion matrix
11. **Export / Load** - signed `.bundle` (see Known limitations)

### ML core (v5.0)
`toolkit/ids_core` is the new, leakage-free pipeline. It has no web or network code; the web trainer uses it since 5.0-rc1 and the CLI since step 0.5.

1. **Load** (`datasets.load_dataset`): encoding and separator sniffed, header spaces stripped, exact dtype downcast, SHA-256; the dataset family (CIC-IDS2017, NSL-KDD or generic) is recognised from the content, not the file name.
2. **Split** (`split.split_dataset`): exact copies removed first, then the profile's split: `official` (NSL-KDD's own files, kept as published), `temporal` / `temporal_per_class` (validated time column), `group`, or `stratified` with an "optimistic" warning. Every fallback has a reason code.
3. **Leakage guard** (`leakage.check_leakage`, training split only): blocks label copies, IDs, IPs and timestamps.
4. **Train** (`pipeline.train_models`): one pipeline per model, fitted on training rows only: impute → encode → scale → select → balance → model. Every seed is fixed and CPU use is capped by `TOOLKIT_MAX_WORKERS`.
5. **Evaluate** (`evaluation.evaluate`): cross-validation on the training split, the model chosen by CV macro-F1 **before** the test split is used, then the test split scored once; macro-F1, balanced accuracy, per-class recall and a leaderboard.

```python
from toolkit.ids_core.datasets import load_dataset
from toolkit.ids_core.split import split_dataset
from toolkit.ids_core.leakage import check_leakage
from toolkit.ids_core.evaluation import evaluate, leaderboard_table

data = load_dataset("data/raw/CIC-2017.csv")
split = split_dataset(data.frame, "Label", profile=data.profile)
guard = check_leakage(split.train, "Label", profile_blocked=data.profile.blocked_columns)
result = evaluate(split.train, split.test, "Label", features=guard.features,
                  models=("random_forest", "xgboost"), balancer="recommended")
print(leaderboard_table(result))
```

---

## 🔑 API Keys Setup

### Optional APIs (for Port Scanner)
The scanner works without any key. CVE lookups use the NVD API, which works without a key but allows more requests with one.

1. **OpenRouter** (for the optional AI explanation)
   - Sign up: https://openrouter.ai/
   - Get API key

2. **AbuseIPDB** (for IP reputation)
   - Sign up: https://www.abuseipdb.com/
   - Get API key

3. **NVD** (optional, faster CVE lookups)
   - Request a key: https://nvd.nist.gov/developers/request-an-api-key

### Configuration
Set environment variables (`OPENROUTER_API_KEY`, `ABUSEIPDB_API_KEY`, `NVD_API_KEY`, optionally `OPENROUTER_MODEL`), or put the same `NAME=value` lines in a `.env` file in the project folder: copy `.env.example` to `.env` and fill in what you use. An environment variable wins over `.env`. The older key file still works as a fallback:

```bash
# Copy the example file
cp tools/Port_scan/api_keys.example.txt tools/Port_scan/api_keys.txt

# Edit with your keys
notepad tools/Port_scan/api_keys.txt
```

Format:
```
openrouter=your_openrouter_api_key_here
abuseipdb=your_abuseipdb_api_key_here
nvd=your_nvd_api_key_here
```

⚠️ **Never commit `api_keys.txt` or `.env` to Git!** (Both are in .gitignore)

### Log files
Each tool writes a log file in `instance/logs/` (`web.log`, `scanner.log`, `ids_cli.log`), UTF-8, rotated at 1 MB with 3 old files kept. Set `TOOLKIT_LOG_LEVEL` (`DEBUG`, `INFO`, `WARNING`, `ERROR`; default `INFO`) as an environment variable to change how much is written. API keys and `Authorization` / `Bearer` values are replaced by `[REDACTED]`, and dataset values are never logged. Library warnings (for example a scikit-learn version mismatch) appear on the console and in the log.

---

## 📸 Screenshots

### Main Menu
```
 _       __       __                            
| |     / /___   / /_____ ____   ____ ___   ___ 
| | /| / // _ \ / // ___// __ \ / __ `__ \ / _ \
| |/ |/ //  __// // /__ / /_/ // / / / / //  __/
|__/|__/ \___//_/ \___/ \____//_/ /_/ /_/ \___/ 

Network Recon and Detection Tool

1. Network Recon
2. Intrusion Detection (CLI)
3. Intrusion Detection (Web)
4. Exit
```

### Port Scan Results
```
+----------------------------------------------------------+
| PORT      | STATE   | SERVICE    | BANNER                 |
+----------------------------------------------------------+
| 22/tcp    | open    | ssh        | OpenSSH 8.2            |
| 80/tcp    | open    | http       | nginx/1.18.0           |
| 443/tcp   | open    | https      | nginx/1.18.0           |
+----------------------------------------------------------+
```

### Web IDS - Data Analysis
```
📊 Data Analysis
┌────────────────────┬─────────────────────────┐
│ Imbalance Ratio    │ 2.2%                    │
│ Severity           │ SEVERE                  │
│ Dataset Size       │ 122,859 (very_large)    │
│ Features           │ 41 (medium)             │
│ Classes            │ 7 (multi-class)         │
│ Smallest Class     │ 2,754 samples           │
└────────────────────┴─────────────────────────┘

✅ Recommended Method: SMOTE-ENN
Large multi-class dataset with severe imbalance. 
SMOTE-ENN cleans noisy synthetic samples.

Alternatives: Class Weights, SMOTE + Tomek, ADASYN
```

### IDS Prediction (output format only, not a real result)
```
================================================================
🎯 PREDICTION RESULT
================================================================
   Status: ✅ BENIGN
   Confidence: 98.50%

   📊 Top Probabilities:
      BENIGN                    98.50% ████████████████████
      Portscan                   0.80%
      DDoS                       0.35%
================================================================
```

---

## 🛠️ Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Port Scanner timeout | Check firewall settings |
| SMOTE fails | Minority class needs ≥6 samples |
| Web app not starting | Check port 5000 is free |

### Memory Issues with Large Datasets
For datasets >500K rows:
1. Use **Class Weights** (no extra memory)
2. Use **Random Undersampling** (reduces data size)
3. Avoid SMOTE variants (create new samples)

---

## 🤝 Contributing

1. Create a branch (`git checkout -b fix/short-name`)
2. Write a test for the correct behaviour first, then make the fix
3. Make sure `python -m pytest` and `python -m ruff check .` pass (the pre-commit hook runs both)
4. Commit with a clear message

---

## 📄 License

Intended licence: **MIT**. The `LICENSE` file has not been added yet (planned in v5.1, see [docs/IMPROVEMENT_PLAN.md](docs/IMPROVEMENT_PLAN.md) D20). Datasets keep their own terms and must be cited when used; see "Licences and citations" in [docs/datasets.md](docs/datasets.md). UNSW-NB15 is free for academic research only.

---

## 🙏 Acknowledgments

- [CIC-IDS-2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [scikit-learn](https://scikit-learn.org/)
- [imbalanced-learn](https://imbalanced-learn.org/)
- [Flask](https://flask.palletsprojects.com/)
- [AbuseIPDB](https://www.abuseipdb.com/)
- [OpenRouter](https://openrouter.ai/)

---

## 👨‍💻 Author

**Ahmed Mohamed Abdalrazek**

- Graduation Project - 2025
- Network Security Toolkit

---

<p align="center">Made with ❤️ for cybersecurity</p>
