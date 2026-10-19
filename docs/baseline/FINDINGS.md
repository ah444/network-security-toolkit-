# Baseline findings — what the "before" numbers show

Hand-written notes on the generated results in [README.md](README.md). Every statement below was checked against the JSON files in this folder. These are observations about the **original** app (commit `da41472`), not the fixed version.

## 1. Results are fully reproducible run to run
All 12 runs (2 datasets × 2 scenarios × 3 runs) gave **identical** metrics, so the spread is 0.00 everywhere. The scenarios use the seeded paths of the app (`random_state=42`, seeded `/apply_smote`). The unseeded inner SMOTE in SMOTE-ENN / SMOTE+Tomek (B20) is **not** exercised by these scenarios, because the app's recommender chose `none` (CIC) and `class_weight` (NSL-KDD). So "deterministic" here does not prove B20 wrong.

## 2. The label-derived NSL-KDD column `level` is used as a feature
- **S1** trains on all 41 remaining columns, **including `level`** (leak L6). `level` is a difficulty score derived from how many classifiers got each record right, i.e. it comes from the labels.
- **S2:** the app's own feature recommender (`/recommend_features`, SelectKBest on the full SMOTE-balanced data) **ranked `level` among the top 10 features** and it was used for training. The recommender actively pushes users towards the leak.

## 3. "Apply SMOTE" creates synthetic test data (leak L2)
- **NSL-KDD S2:** the "Apply SMOTE" button grew the dataset from **125,973 to 947,922 rows** *before* the train/validation/test split. The 142,189-row test set therefore consists mostly of SMOTE-generated rows (about 87 % of all rows are synthetic). S2's ~81 % accuracy is measured on **synthetic test data**, which says little about real traffic.
- **CIC DDoS S2:** here "Apply SMOTE" first downsampled the larger class (DDoS) to the size of BENIGN (225,745 → 195,436 rows), and SMOTE then had nothing left to add. No synthetic rows, but 30,309 real DDoS rows were discarded before the split, and feature selection still ran on the full dataset (L3).
- Because the S2 test sets differ in size and content from S1's, **S1 and S2 numbers are not directly comparable** with each other. Each is a baseline for its own click path.

## 4. Accuracy hides the minority classes (L9)
On NSL-KDD S1, random forest scores **99.82 % accuracy / 99.80 % weighted F1** but only **76.62 % macro-F1 and 74.94 % balanced accuracy**. The common classes dominate accuracy, while rare attack types (several with fewer than 10 test rows) are often missed. The app only shows accuracy and weighted F1, so users never see this.

## 5. Logistic regression with class weights collapses on NSL-KDD S1
With the recommended `class_weight` balancing, logistic regression reaches only **8.44 % accuracy** (7.76 % macro-F1). It predicts the rare class `satan` for 6,786 of the 18,896 test rows. The balanced class weights over 23 very unequal classes (the smallest has 2 samples) overwhelm the linear model. The app reported this result without any warning.

## 6. CIC DDoS is almost trivially separable
Tree models reach **99.9 %** on the binary BENIGN/DDoS file with 14 features. This matches the plan's note under the leakage guard: single packet-size features can separate these two classes almost perfectly, which is why the v5.0 guard only *warns* about strong continuous features instead of blocking them. 23,416 duplicate rows (L4) in this file also sit on both sides of the random split; their effect is measured in v5.0 by comparing with a deduplicated split.

## 7. The v4 CLI model is driven by the leaked `Attempted Category` input (L1)
Source: [`cli_v4.json`](cli_v4.json). The v4 test data isn't available, so accuracy can't be re-measured. Instead the CLI's own 4 built-in presets were predicted, and then re-predicted with **only** `Attempted Category` changed.
- **None of the 4 presets is classified as what it is meant to be.** With the shipped value `Attempted Category = 0.0`, both models label every preset as an "**- Attempted**" class. The "BENIGN-like" preset becomes `DoS Slowhttptest - Attempted`, and the "DDoS-like" and "PortScan-like" presets become `SSH-Patator - Attempted` / `DoS Slowloris - Attempted`.
- **Changing only that one input flips the result.** At `Attempted Category = -1`, every preset, including the DDoS-like and PortScan-like ones, becomes **BENIGN** with 0.88–1.00 confidence. At `0`, `1` or `2` it becomes various "Attempted" classes. The real traffic features barely matter.
- So the v4 model mostly reads the answer from a field derived from the label. This confirms the warning in `models/README.md` with direct evidence, and it means the CLI's ~97–98 % accuracy claim says nothing about detecting real traffic.
- Loading the v4 files on this machine raises `InconsistentVersionWarning` (pickled with scikit-learn **1.6.1**, loaded with **1.8.0**) for the scaler, label encoder, random forest and every tree, plus an XGBoost serialization warning. The CLI hides all of them with its global `filterwarnings` (B18, B19).

## What v5.0 must show against this baseline
- The same datasets through the fixed pipeline: deduplicated, split before any SMOTE or feature selection, `level` blocked, macro-F1 and balanced accuracy reported.
- Expected direction: lower headline accuracy on NSL-KDD (no `level`, no synthetic test rows) and a smaller gap between accuracy and macro-F1 being visible to the user. The size of any change will be measured, not assumed.
