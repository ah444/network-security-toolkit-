// Display texts: numbers, balancing method names and hints, split reason/warning codes.

// Number for display: fixed decimals, or "n/a" when missing
export function fmt(value, digits) {
  return value === null || value === undefined || isNaN(value) ? "n/a" : Number(value).toFixed(digits);
}

// A count with thousands separators (125,973), the same in every browser language
export function wholeNumber(value) {
  return Number(value).toLocaleString("en-US");
}

// Feature score for display: the server sends "inf" for a feature with one value per class
export function scoreText(score) {
  return score === "inf" ? "∞" : fmt(score, 2);
}

export const INFINITE_SCORE_HINT = "∞: the feature has one value per class on this split, so it separates the classes perfectly. Check that it isn't a label leak.";

// Imbalance ratio (smallest / largest class) for display
export function ratioText(ratio) {
  const percent = ratio * 100;
  if (percent >= 1) {
    return percent.toFixed(1) + "%";
  } else if (percent >= 0.01) {
    return percent.toFixed(2) + "%";
  } else if (percent > 0) {
    return percent.toExponential(2) + " (1:" + Math.round(1 / ratio) + ")";
  }
  return "0%";
}

export const BALANCING_NAMES = {
  "none": "None (No Balancing)",
  "recommended": "Recommended",
  "class_weight": "Class Weights",
  "smote": "SMOTE",
  "smote_enn": "SMOTE-ENN",
  "adasyn": "ADASYN",
  "borderline_smote": "Borderline-SMOTE",
  "random_undersample": "Random Undersampling",
  "tomek": "Tomek Links",
  "smote_tomek": "SMOTE + Tomek Links"
};

export function balancingName(method) {
  return BALANCING_NAMES[method] || method;
}

// Hint under the balancing method list; { strong: text } parts are shown in bold
export const BALANCING_HINTS = {
  "none": "No balancing will be applied. Use this if your dataset is already balanced.",
  "recommended": ["The toolkit picks a method from the ", { strong: "training split's" }, " class counts when you train."],
  "class_weight": "Fast and memory-efficient. Adjusts model weights to give more importance to minority classes.",
  "smote": "Creates synthetic samples for minority classes. Good for continuous features. May increase training time.",
  "smote_enn": "SMOTE + Edited Nearest Neighbors. Creates synthetic samples then removes noisy ones.",
  "adasyn": "Adaptive SMOTE - focuses on harder-to-learn samples.",
  "borderline_smote": "Only generates synthetic samples near class boundaries. Good when classes overlap.",
  "random_undersample": "Removes random samples from bigger classes. Very fast but may lose important data.",
  "tomek": "Removes overlapping samples between classes. Cleans decision boundaries without adding data.",
  "smote_tomek": "SMOTE + Tomek Links. Creates synthetic samples then cleans boundaries."
};

// Plain-language text for the split's reason and warning codes (toolkit/core/services/ids/split.py)
const SPLIT_MESSAGES = {
  "STRATIFIED_SCORES_MAY_BE_OPTIMISTIC": "Rows were split at random (stratified by class), so flows of the same attack can be on both sides: the scores may be optimistic.",
  "CLASS_MISSING_FROM_TEST": "Some classes have no rows in the test split, so they are not tested.",
  "CLASS_MISSING_FROM_TRAIN": "Some classes appear only in the test split; no model can learn them.",
  "OFFICIAL_TEST_OVERLAP_KEPT": "Some rows of the official test file also occur in the training file. They are kept as published, so results stay comparable with other work.",
  "OFFICIAL_COPIES_KEPT": "The official files contain repeated rows; they are kept as published.",
  "OFFICIAL_TEST_OVERLAP_REMOVED": "Test rows that also occur in the training file were removed (strict mode).",
  "TIME_MISSING_ROWS_DROPPED": "Rows without a time value were left out of the time-ordered split.",
  "LABEL_MISSING_ROWS_DROPPED": "Rows without a label were left out.",
  "OFFICIAL_TEST_FILE_MISSING": "No official test file was uploaded.",
  "TIME_COLUMN_MISSING": "There is no usable time column.",
  "TIME_MISSING_RATIO_EXCEEDED": "Too many rows have no time value.",
  "TIME_PARSE_FAILURE_RATIO_EXCEEDED": "Too many time values could not be read.",
  "TIME_FORMAT_AMBIGUOUS": "The date format is ambiguous (day/month order).",
  "TIME_NON_MONOTONIC_ACROSS_FILES": "The files' time ranges overlap or go backwards.",
  "TIME_TOO_COARSE_FOR_CLASS": "A class happens within too short a time to split by time.",
  "GROUP_COLUMN_MISSING": "There is no group column.",
  "GROUP_TOO_FEW_GROUPS": "There are too few groups to split by group.",
  "GROUP_CLASS_ONE_SIDED": "A class occurs in one group only."
};

export function splitMessage(code) {
  return SPLIT_MESSAGES[code] || code;
}
