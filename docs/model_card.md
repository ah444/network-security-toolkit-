# Model card: the web trainer's intrusion-detection models

No trained model ships with the project. You train models in the web trainer and export
them as signed bundles. This card describes those models: what they are for, how they are
built, and how well they did in the project's own measurements, the ML report in
[`docs/report/`](report/FINDINGS.md). Every number below is copied from that report
(`tests/test_docs_architecture.py` checks it).

Each bundle also carries its own card, with the numbers of the run that made it:

```
python -m toolkit ids --bundle instance/bundles/<run id>.bundle --info
```

## Model details

- **Type:** supervised multi-class classifiers of network flows (one row = one flow or
  connection record), from scikit-learn, imbalanced-learn and, with the `[boost]` extra,
  XGBoost and LightGBM. Eleven algorithms; random forest and logistic regression are the
  defaults ([architecture](architecture.md#5-registries)).
- **Pipeline:** missing and infinite values imputed, optional signed log and scaling
  (linear and distance models), categorical columns one-hot or ordinal, optional feature
  selection and resampling, then the model. Every step is fitted on the training split only.
- **Model choice:** by cross-validated macro-F1 on the training split; the test split is
  scored once, after the choice.
- **Output:** a class label per row and, for models that have them, class probabilities.
- **Bundle:** the fitted models plus the full TrainingRun record (data fingerprint, settings,
  split, leakage findings, scores, library versions), signed with this install's key. A
  bundle loads only on the install that made it.

## Intended use

- Learning how IDS models behave and comparing algorithms, balancing methods and feature
  sets on the public datasets (CIC-IDS2017, NSL-KDD) or on flows from your own network.
- Labelling flow files offline (`toolkit ids --csv`) to see what a model would flag.

**Out of scope:** blocking or allowing live traffic, any automatic response, or production
intrusion prevention. The models see one flow at a time, offline, and have not been tested
against an attacker who adapts to them.

## Training data

The datasets are not part of the project; their licences, citations and redistribution
terms are in [datasets.md](datasets.md).

- **CIC-IDS2017** (DDoS day file in the report): binary BENIGN / DDoS. 23,416 exact duplicate
  rows are removed before the split.
- **NSL-KDD:** `KDDTrain+` (stratified split), or `KDDTrain+` against the official
  `KDDTest+` file. The label-derived `level` column is blocked by the leakage guard.

## Metrics

Macro-F1 and balanced accuracy lead: they weigh every attack class equally, so a model that
misses the rare attacks can't hide behind the majority class. Accuracy, weighted F1,
per-class recall and the confusion matrix are shown too.

## Results (from the ML report)

Test split, mean of 3 runs (all 3 identical: everything is seeded), Windows, one core,
scikit-learn 1.8.0.

| Case | Best result | What it shows |
|---|---|---|
| CIC-IDS2017 DDoS, all features | random forest and decision tree 99.88–99.91 % macro-F1; logistic regression 98.05 % | this file is almost perfectly separable; the split is random, so scores may be optimistic |
| NSL-KDD `KDDTrain+` only, all features | random forest 99.82 % accuracy but 68.75 % macro-F1; decision tree 77.93 % macro-F1 | accuracy hides the rare classes |
| NSL-KDD `KDDTrain+` only, SMOTE + 10 features | random forest 44.33 % macro-F1, 71.45 % accuracy | this combination is a poor choice on NSL-KDD |
| NSL-KDD official `KDDTest+` | decision tree 72.13 % accuracy, 26.96 % macro-F1, 32.23 % balanced accuracy | the realistic NSL-KDD numbers |

On the official test file, 17 attack types never appear in `KDDTrain+`; they are 3,750 test
rows (16.6 %), and no classifier can predict a class it never saw. On the test rows of known
classes, about 86.5 % are correct. Details and the per-model table:
[FINDINGS.md](report/FINDINGS.md) and [README.md](report/README.md).

The numbers were measured with the 5.0 ML core. The v5.1 changes to `toolkit/ids_core/`
(progress reporting, logging, batch prediction, type annotations) don't change how models
are trained, and the report has not been re-run for them.

## Limitations

- **Splits:** without a time column the split is random (stratified), so flows of one attack
  can be on both sides; the run records the warning `STRATIFIED_SCORES_MAY_BE_OPTIMISTIC`.
- **Rare classes** with one or two test rows are often missed (recall 0).
- **New attacks:** a class that is not in the training data is always predicted wrongly.
- **Columns:** a bundle predicts only rows that have the feature columns it was trained on,
  with the same meaning (the same flow tool and settings).
- **Other networks:** scores on the public datasets don't carry over to other traffic.
- **Probabilities:** a random forest's probabilities can change in their last digits from
  one run to the next (its trees are summed in parallel); the CSV output rounds them to 4
  digits.

## Ethical considerations

- The data stays on this computer: training, prediction and bundles make no network request.
- The toolkit's port scanner can be used against systems without permission. It asks for
  confirmation that you are authorised to scan a target, and it should be used only on your
  own systems or with written permission.
- A model's "attack" label is a guess with known error rates (see above), not proof of an
  intrusion; don't act against a person or system on its output alone.
- The public datasets have their own terms, including for models derived from them
  ([datasets.md](datasets.md)).
