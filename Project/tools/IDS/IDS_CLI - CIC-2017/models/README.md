# 🛡️ IDS Models - CIC-IDS2017

## Current Version: 4.0

### Files Included:
- `config.json` - Model configuration (features, classes, best model)
- `scaler.pkl` - RobustScaler for feature normalization
- `label_encoder.pkl` - LabelEncoder for class labels
- `random_forest.pkl` - Random Forest classifier model
- `xgboost.pkl` - XGBoost classifier model

### Features Used (8 features):
1. Attempted Category
2. Bwd Segment Size Avg
3. Bwd Packet Length Mean
4. Bwd Packet Length Std
5. Bwd Packet Length Max
6. Subflow Bwd Bytes
7. Packet Length Std
8. Packet Length Mean

### Classes Detected (16 classes):
- BENIGN
- Botnet / Botnet - Attempted
- DDoS
- DoS GoldenEye
- DoS Hulk / DoS Hulk - Attempted
- DoS Slowhttptest / DoS Slowhttptest - Attempted
- DoS Slowloris / DoS Slowloris - Attempted
- FTP-Patator / FTP-Patator - Attempted
- Portscan
- SSH-Patator / SSH-Patator - Attempted

---

## ⚠️ Important Note on Model Version

### Current Models (v4.0)
The current models include `Attempted Category` as a feature. This feature is derived from the label naming convention in the dataset and may provide artificially high accuracy.

### Recommended: Version 5.0
For production use, regenerate models using `notebook_v5_fixed.ipynb` which:
- ✅ Excludes label-derived features (no data leakage)
- ✅ Uses only real packet features extractable from network traffic
- ✅ Provides more realistic accuracy expectations (85-92%)

---

## How to Regenerate Models

### Option 1: Using Jupyter Notebook (Recommended)
```bash
# Open the notebook
jupyter notebook notebook_v5_fixed.ipynb

# Run all cells to generate new models
```

### Option 2: Using Google Colab
1. Upload `notebook_v5_fixed.ipynb` to Google Colab
2. Upload your Kaggle API key (`kaggle.json`)
3. Run all cells
4. Download the generated `models/` folder

### Dataset Required
- **Source**: [Kaggle - Improved CIC-IDS 2017/2018](https://www.kaggle.com/datasets/ernie55ernie/improved-cicids2017-and-csecicids2018)
- **Size**: ~10GB (includes both 2017 and 2018 datasets)

---

## Model Performance

### Random Forest (Best Model)
| Metric | Score |
|--------|-------|
| Train Accuracy | ~99% |
| Test Accuracy | ~98% |
| F1 Score (weighted) | ~98% |

### XGBoost
| Metric | Score |
|--------|-------|
| Train Accuracy | ~99% |
| Test Accuracy | ~97% |
| F1 Score (weighted) | ~97% |

*Note: Accuracy is high due to `Attempted Category` feature. Expect 85-92% with v5.0 models.*

---

## Usage Example

```python
from IDS_with_Machine_Learning import IDSModel

# Initialize and load model
ids = IDSModel()
ids.load('random_forest')

# Make prediction
features = {
    "Attempted Category": 0.0,
    "Bwd Segment Size Avg": 156.5,
    "Bwd Packet Length Mean": 312.4,
    "Bwd Packet Length Std": 425.8,
    "Bwd Packet Length Max": 1460.0,
    "Subflow Bwd Bytes": 4892.0,
    "Packet Length Std": 498.2,
    "Packet Length Mean": 245.6
}
result = ids.predict(features_dict=features)
print(f"Prediction: {result['label']} (Confidence: {result['confidence']:.2%})")
```

---

## Training Details

- **Dataset**: CIC-IDS-2017 (improved version)
- **Train/Val/Test Split**: 70% / 15% / 15%
- **Scaling**: RobustScaler
- **Feature Selection**: SelectKBest with F-test
- **Cross-Validation**: 5-fold StratifiedKFold
