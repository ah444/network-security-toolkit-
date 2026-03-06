# 🛡️ Network Security Toolkit

A comprehensive network security tool combining **Network Reconnaissance** and **Machine Learning-based Intrusion Detection System (IDS)**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.9%2B-red)

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
- 🔐 CVE vulnerability lookup (via MITRE)
- 🤖 AI-powered vulnerability analysis (via DeepSeek)
- 📊 IP reputation check (via AbuseIPDB)
- 📁 Export results to JSON/CSV/TXT

### 2. Intrusion Detection System (CLI)
- 🧠 Machine Learning-based detection
- 📊 Random Forest & XGBoost models
- 🎯 16 attack classes detection
- ⚡ Real-time prediction
- 📈 High accuracy (~97%)

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
- 💾 **Export Models** - Save trained models as .pkl

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/network-security-toolkit.git
cd network-security-toolkit

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
flask>=2.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.9.0
requests>=2.26.0
dask>=2022.1.0
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
- Enter target IP or domain
- Specify port range
- Choose TCP/UDP/both protocols
- View vulnerability reports
- Export results

### IDS CLI
```bash
# Run directly
python "tools/IDS/IDS_CLI - CIC-2017/IDS_with_Machine_Learning.py"
```

Features:
- Load pre-trained models
- Manual feature input
- Sample data testing
- Model switching

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
The system automatically analyzes your dataset:

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
- Select trained model (Random Forest, KNN, Logistic Regression)
- Choose dataset (Test or Validation)
- View matrix with color-coded cells
- See True Positives, False Positives, etc.

### 🔀 Train/Test Split Slider
Customize data splits:
- Slider range: 10% to 50% for (Validation + Test)
- Example: 30% → Train: 70% | Val: 15% | Test: 15%
- Real-time label update

---

## 📁 Project Structure

```
Project/
├── main.py                          # Main entry point
├── requirements.txt                 # Dependencies
├── .gitignore                       # Git ignore rules
├── README.md                        # This file
│
└── tools/
    ├── Port_scan/                   # Network Reconnaissance
    │   ├── Port_Scan.py             # Port scanner
    │   └── api_keys.example.txt     # API keys template
    │
    └── IDS/
        ├── IDS_CLI - CIC-2017/      # CLI-based IDS
        │   ├── IDS_with_Machine_Learning.py
        │   └── models/
        │       ├── config.json      # Model configuration
        │       ├── scaler.pkl       # RobustScaler
        │       ├── label_encoder.pkl
        │       ├── random_forest.pkl
        │       ├── xgboost.pkl
        │       └── README.md
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
            └── models/              # Trained models (.pkl)
```

---

## 🤖 Machine Learning Models

### Supported Datasets
- **CIC-IDS-2017** - Canadian Institute for Cybersecurity
- **CIC-IDS-2018** - Updated version
- **NSL-KDD** - Network intrusion dataset
- **UNSW-NB15** - Australian dataset
- **Custom CSV/TXT** - Any properly formatted dataset

### Algorithms
| Algorithm | Type | Strengths |
|-----------|------|-----------|
| **Random Forest** | Ensemble | High accuracy, robust |
| **Logistic Regression** | Linear | Fast, interpretable |
| **K-Nearest Neighbors** | Instance-based | Simple, no training |

### Attack Classes Detected
- BENIGN (Normal traffic)
- Botnet
- DDoS
- DoS (GoldenEye, Hulk, Slowhttptest, Slowloris)
- FTP-Patator
- SSH-Patator
- Portscan
- Web Attacks
- Infiltration
- Heartbleed
- And more...

### Training Pipeline
1. **Data Upload** - CSV/TXT with headers
2. **Preview** - View data statistics
3. **Class Analysis** - Distribution & imbalance detection
4. **Filter Classes** - Remove unwanted targets (optional)
5. **Feature Selection** - SelectKBest (k configurable)
6. **Balancing** - 8 methods available
7. **Split** - Train/Val/Test (customizable)
8. **Training** - Multiple algorithms
9. **Evaluation** - Accuracy, Confusion Matrix
10. **Export** - Save models as .pkl

---

## 🔑 API Keys Setup

### Required APIs (for Port Scanner)
1. **OpenRouter** (for AI vulnerability analysis)
   - Sign up: https://openrouter.ai/
   - Get API key

2. **AbuseIPDB** (for IP reputation)
   - Sign up: https://www.abuseipdb.com/
   - Get API key

### Configuration
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
```

⚠️ **Never commit `api_keys.txt` to Git!** (It's in .gitignore)

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

### IDS Prediction
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

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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
"# network-security-toolkit-" 
"# network-security-toolkit-" 
