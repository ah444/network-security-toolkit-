# 🛡️ IDS Models - CIC-IDS2017 (v4)

## ⚠️ Status: leaky model, kept only until v5.0 replaces it

The v4 model uses **`Attempted Category`** as an input feature. That value comes from the attack **label** itself (the "… - Attempted" classes), so the model can read the answer from its input. A measured baseline ([docs/baseline/FINDINGS.md](../../../../docs/baseline/FINDINGS.md), section 7; data in [docs/baseline/cli_v4.json](../../../../docs/baseline/cli_v4.json)) shows:

- **None of the CLI's 4 built-in sample flows is classified as intended.** With `Attempted Category = 0.0` (the value the samples use), both models predict an "… - Attempted" class for every sample, including the "BENIGN-like" one.
- **Changing only `Attempted Category` flips the result.** At `-1`, every sample (including the DDoS-like and PortScan-like ones) becomes BENIGN with 0.88–1.00 confidence.
- **The accuracy figures previously listed here (~97–98 %) are withdrawn.** They came from evaluating with this leaked feature, and the original test data isn't available to re-measure them.

The CLI prints a warning when this model is loaded. A replacement is planned in v5.0: it is trained in the fixed web trainer (leakage guard on, split before any balancing or feature selection) and exported as a signed bundle (see [docs/IMPROVEMENT_PLAN.md](../../../../docs/IMPROVEMENT_PLAN.md), D12).

### Version note
The `.pkl` files were saved with **scikit-learn 1.6.1** (and numpy 2.x). Loading them with a different scikit-learn version raises `InconsistentVersionWarning` and may give wrong results. The CLI currently hides these warnings; this is fixed in v5.0 step 0.3.

---

## Files Included
- `config.json` - Model configuration (features, classes, best model)
- `scaler.pkl` - RobustScaler for feature normalization
- `label_encoder.pkl` - LabelEncoder for class labels
- `random_forest.pkl` - Random Forest classifier model
- `xgboost.pkl` - XGBoost classifier model

## Features Used (8 features)
1. Attempted Category  ⚠️ label-derived (see above)
2. Bwd Segment Size Avg
3. Bwd Packet Length Mean
4. Bwd Packet Length Std
5. Bwd Packet Length Max
6. Subflow Bwd Bytes
7. Packet Length Std
8. Packet Length Mean

## Classes (16 classes, "improved" CIC-IDS2017 labelling)
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

## Training details (as originally documented, not verifiable)

The training code (`notebook_v5_fixed.ipynb`, referenced in earlier versions of this file) is **not in the project**, so the original training run can't be reproduced or checked. The original documentation stated:

- Dataset: improved CIC-IDS2017 / CSE-CIC-IDS2018 (Kaggle), see `config.json` → `datasets`
- Train/Val/Test split 70 / 15 / 15, RobustScaler, SelectKBest (F-test), 5-fold StratifiedKFold

The dataset available locally is the **original** CIC-IDS2017 variant (15 classes, no `Attempted Category`; see [docs/datasets.md](../../../../docs/datasets.md)), so v5.0 does not retrain this exact model. It trains a new one through the fixed pipeline, and every training run is recorded (dataset hash, split, features, versions, metrics).

---

## Usage Example

```python
from IDS_with_Machine_Learning import IDSModel

ids = IDSModel()
ids.load('random_forest')   # prints the leakage warning for this v4 model

features = {
    "Attempted Category": 0.0,   # label-derived: this value largely decides the output
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
