# ML report findings — v4 baseline vs the v5.0 ML core

Hand-written notes on the generated table in [README.md](README.md). Every number below was
checked against the JSON files in this folder and in [`docs/baseline/`](../baseline/README.md).
"v4" is the original web trainer (commit `da41472`, measured in
[baseline FINDINGS](../baseline/FINDINGS.md)); "v5" is the fixed ML core
(`toolkit.ids_core`), which the web trainer uses since 5.0-rc1. All numbers are test-split
scores, mean of 3 runs, on one core.

## 0. Read this first: the two columns don't measure the same thing
v4 and v5 were run on the same files with the same models and the same intent per scenario,
but **v4's test sets were not valid**, so a v5 number lower than v4's is not a regression.
What changed between them:

| | v4 | v5 |
|---|---|---|
| Duplicate rows | kept, can be on both sides of the split | removed before the split (CIC: 23,416 of 225,745 rows) |
| Split | random 70/15/15, not stratified | stratified 80/20; NSL-KDD official case: `KDDTrain+` / `KDDTest+` |
| SMOTE (S2) | on the whole file **before** the split: NSL-KDD test set ~87 % synthetic rows | inside the pipeline, on the training split only; test rows are all real |
| Feature selection (S2) | on the whole SMOTE-balanced file | inside the pipeline, on the real training rows only |
| NSL-KDD `level` (label-derived) | used as a feature in S1, and ranked top-10 by the v4 recommender in S2 | blocked by the leakage guard |
| Model choice | by test score | by 3-fold cross-validation on the training split; the test split is used only after the choice |
| Metrics shown | accuracy, weighted F1 | also macro-F1, balanced accuracy, per-class recall |

## 1. Results are reproducible
All three runs of every case gave **identical** metrics (spread 0.00 everywhere) and
chose the same model by cross-validation. The seed is fixed and every seeded estimator
gets it (`tests/ml/test_pipeline.py::test_identical_runs_identical_results`).

## 2. CIC-IDS2017 DDoS: tree scores unchanged, logistic regression much better
- Random forest and decision tree stay at **99.88–99.91 %** macro-F1 in both scenarios
  (v4: 99.91–99.92 %). Removing the 23,416 duplicate rows and using real test rows only
  did not lower them: this binary BENIGN/DDoS file is almost perfectly separable by
  packet-size features. The leakage guard warns about two of them (`Bwd Packet Length
  Mean`, `Avg Bwd Segment Size`) but, as designed, does not block them.
- Logistic regression rises from **95.13 % to 98.05 %** macro-F1 in S1 and from
  **79.97 % to 98.55 %** in S2. v5 gives linear models a signed log transform and standard
  scaling of the heavy-tailed flow features (v4 used `RobustScaler` only).
- The split is still random (stratified), because this file has no time column; flows of
  one attack can be on both sides, so these scores may still be optimistic. v5 records
  this as the warning `STRATIFIED_SCORES_MAY_BE_OPTIMISTIC`.

## 3. NSL-KDD (`KDDTrain+` only), S1: accuracy unchanged, macro-F1 now visible
- Random forest keeps **99.82 % accuracy** but has **68.75 % macro-F1** (v4: 76.62 %). Without
  `level`, the rare classes are missed more often: in the test split, `ftp_write`,
  `rootkit`, `loadmodule`, `multihop`, `perl` and `spy` (1–2 test rows each) have recall 0
  for random forest. The 99.82 % accuracy comes from `normal` and `neptune`, which make up
  86 % of the test rows.
- Decision tree reaches **77.93 %** macro-F1 on the test split, higher than random forest,
  but cross-validation on the training split picked random forest (CV macro-F1 74.15 %
  ± 1.24 vs 73.10 % ± 3.68). v5 does not switch to the test-best model: choosing by test
  score is exactly the selection bias v4 had (L8).
- Logistic regression, which collapsed in v4 (**7.76 %** macro-F1, 8.44 % accuracy), now
  reaches **67.70 %** macro-F1, **97.19 %** accuracy and **84.65 %** balanced accuracy with the
  same `class_weight` balancing (scaled inputs, one-hot encoded `protocol_type`, `service`,
  `flag`).

## 4. NSL-KDD (`KDDTrain+` only), S2: the v4 score came from synthetic test rows and `level`
- v4's S2 reported about **81 %** on every metric. Its test set was ~87 % SMOTE-generated
  rows and its 10 features included `level` (baseline FINDINGS §2–3), so that number does not
  describe real traffic.
- v5 S2 scores **44.33 %** macro-F1 and **71.45 %** accuracy (random forest). An extra
  diagnostic run (same split and the same 10 F-test features, **without** SMOTE, random
  forest + decision tree, one run, not part of the report) gave **47.6 %** macro-F1 and
  **95.4 %** accuracy, with `normal` recall 0.99. So:
  - the low macro-F1 comes from **keeping only 10 F-test features** (without `level`, they
    don't separate `smurf`, `back` and `warezclient`: recall 0.13–0.25), and
  - **SMOTE** then lowers accuracy further: it grows every attack class with at least 2
    training rows to the size of `normal`, and with only 10 features the synthetic attack rows overlap normal traffic,
    so `normal` recall drops from 0.99 to **0.53** (random forest).
- Lesson for users: on NSL-KDD, "SMOTE + 10 best features" is a bad combination. S1 (all
  41 features, class weights) is far better on real test rows.

## 5. NSL-KDD with the official test file: the realistic numbers
v4 had no way to use `KDDTest+` (it has no header row), so there is no v4 column here. v5
trains on all of `KDDTrain+` and scores the official `KDDTest+` (22,544 rows).
- **S1:** cross-validation picked decision tree (CV macro-F1 76.15 % ± 2.41). On the
  official test file it scores **72.13 % accuracy, 26.96 % macro-F1, 32.23 % balanced
  accuracy**; random forest and logistic regression are close (72.22 % / 69.49 % accuracy).
- The main reason is by design of the dataset: **17 attack types in `KDDTest+` never appear
  in `KDDTrain+`** (e.g. `mscan`, `apache2`, `processtable`, `snmpguess`, `saint`). They
  make up **3,750 test rows (16.6 %)**, and a classifier can't predict a class it never
  saw, so all of them are wrong and each adds a 0 to the macro average over 40 classes.
  v5 reports this (`CLASS_MISSING_FROM_TRAIN`, `classes_only_in_test`).
- On the 18,794 test rows of classes the models were trained on, about **86.5 %** are
  correct (decision tree 86.53 %, random forest 86.63 %, computed from the per-class recall
  and support in the JSON). `normal`, `neptune`, `satan` and `smurf` keep recall 0.97–1.0,
  but `guess_passwd` (1,231 rows) and `warezmaster` (944 rows) are almost never found
  (recall 0.14 and 0.003 for decision tree): their test rows look different from the
  training rows.
- **S2** (SMOTE + 10 F-test features) is much worse here too: **42.04 %** accuracy and
  **14.73 %** macro-F1 (decision tree), with `normal` recall 0.42, the same effect as in §4.
- This case is also the slowest: one S2 run took about 50 minutes on one core (three runs
  side by side), because SMOTE grows the attack classes to the size of `normal` (67,343
  rows each) before the models are trained.
- These are the numbers to quote for NSL-KDD: about **72 %** accuracy and **27 %** macro-F1
  on the official test file, not the 99.8 % of a random split of the training file.

## 6. What this means for the claims
- No v5 number is measured on synthetic or duplicated test rows, and no model sees `level`.
- On CIC-IDS2017 DDoS the high scores hold up after deduplication, with the caveat of the
  random split. On NSL-KDD, **accuracy hides the rare classes**: macro-F1 and balanced
  accuracy are the numbers to quote, and the official split (§5) is the realistic one.
- Spread across runs is 0.00 because everything is seeded; it shows reproducibility, not
  a confidence interval. The CV standard deviation in the JSON is the variability across
  folds.
