// Explore, Clean and Features: the target column, class distribution and the balancing
// recommendation, the class filter, the balancing preview and the feature scores. All of
// them use the training split only.

import { getJSON, send } from "./api.js";
import { byId, checkedValues, el, fillCheckboxList, fillSelect, message, setChecked, setContent, table } from "./dom.js";
import { BALANCING_HINTS, INFINITE_SCORE_HINT, balancingName, ratioText, scoreText } from "./format.js";
import { emit, on, state, testSize } from "./state.js";

function showBalancingHint(method, fallback) {
  const hint = BALANCING_HINTS[method];
  if (!hint) {
    setContent(byId("balancingHint"), fallback);
  } else if (typeof hint === "string") {
    setContent(byId("balancingHint"), hint);
  } else {
    setContent(byId("balancingHint"), hint.map(part => part.strong ? el("strong", {}, part.strong) : part));
  }
}

// Classes that can be removed, smallest first
function populateTargetValueFilter(classCounts) {
  const sortedClasses = Object.entries(classCounts).sort((a, b) => a[1] - b[1]);
  fillCheckboxList(byId("targetValuesToRemove"), sortedClasses.map(
    ([className, count]) => ({ value: className, text: className + " (" + count.toLocaleString() + " samples)" })));
}

// The class filter's list for the chosen target
async function loadClassFilter() {
  setContent(byId("targetValuesToRemove"));
  if (!state.target) {
    return;
  }
  try {
    const result = await getJSON("/api/v1/dataset/classes", { target: state.target });
    populateTargetValueFilter(result.value_counts);
  } catch (error) {
    message(byId("targetFilterResult"), "error", "❌ Error: " + error.message);
  }
}

// The target chosen in Explore is used by the class filter and the balancing preview, and
// pre-filled in Features and Train (still changeable there)
function chooseTarget() {
  state.target = byId("targetForDist").value || null;
  if (state.target) {
    byId("targetForRec").value = state.target;
  }
  setContent(byId("classDistributionResult"));
  setContent(byId("targetFilterResult"));
  loadClassFilter();
  emit("target:chosen", state.target);
}

function analysisItem(label, value, valueClass = "") {
  return el("div", { class: "analysis-item" },
    el("span", { class: "analysis-label" }, label), " ",
    el("span", { class: "analysis-value " + valueClass }, value));
}

function recommendationBox(result) {
  const rec = result.recommendation;
  return el("div", { class: "recommendation-box" },
    el("h4", {}, "📊 Training split analysis"),
    el("p", { class: "section-description" },
       `Computed on the ${rec.train_rows.toLocaleString()} training rows only (${result.split.strategy} split); the test rows never influence this choice.`),
    el("div", { class: "analysis-grid" },
      analysisItem("Imbalance Ratio:", ratioText(rec.imbalance_ratio)),
      analysisItem("Severity:", rec.severity.toUpperCase(), "severity-" + rec.severity),
      analysisItem("Training rows:", `${rec.train_rows.toLocaleString()} (${rec.size})`),
      analysisItem("Features:", `${rec.features} (${rec.dimensionality} dimensionality)`),
      analysisItem("Classes:", `${rec.classes} ${rec.classes > 2 ? "(multi-class)" : "(binary)"}`),
      analysisItem("Smallest Class:", `${rec.minority_count.toLocaleString()} samples`,
                   rec.minority_count < 6 ? "severity-extreme" : "")),
    el("div", { class: "recommendation-section" },
      el("h4", {}, "✅ Recommended Method: ", el("span", { class: "recommended-method" }, balancingName(rec.recommended))),
      el("p", { class: "recommendation-reason" }, rec.reason),
      rec.alternatives && rec.alternatives.length > 0 &&
        el("p", { class: "alternatives" }, el("strong", {}, "Alternatives:"), " " + rec.alternatives.map(balancingName).join(", "))));
}

// The recommended balancing method is pre-selected in Clean
async function analyzeClasses(event) {
  event.preventDefault();
  const resultDiv = byId("classDistributionResult");
  const target = state.target;
  if (!target) {
    setContent(resultDiv, "Please select a target column for class distribution analysis.");
    return;
  }
  setContent(resultDiv, "Analyzing class distribution...");
  try {
    const result = await getJSON("/api/v1/dataset/distribution", { target: target, test_size: testSize() });
    byId("balancingMethod").value = result.recommendation.recommended;
    showBalancingHint(result.recommendation.recommended, "");
    setContent(resultDiv,
      el("strong", {}, "Class Distribution (whole dataset):"), el("br"),
      el("div", { class: "table-container" }, el("table", {}, el("tbody", {},
        el("tr", {}, el("th", {}, "Class"), el("th", {}, "Count"), el("th", {}, "Percentage")),
        result.sorted_classes.map(([className, percentage]) => el("tr", {},
          el("td", {}, className), el("td", {}, result.class_counts[className].toLocaleString()),
          el("td", {}, percentage.toFixed(2) + "%")))))),
      el("p", {}, "Total samples: " + result.total_samples.toLocaleString()),
      result.test_file_counts &&
        el("p", {}, "Official test file: " + Object.entries(result.test_file_counts).map(([c, n]) => c + " " + n).join(", ")),
      recommendationBox(result));
  } catch (error) {
    resultDiv.textContent = "Error: " + error.message;
  }
}

async function removeClasses() {
  const selectedValues = checkedValues(byId("targetValuesToRemove"));
  const resultDiv = byId("targetFilterResult");
  if (selectedValues.length === 0) {
    message(resultDiv, "warning", "⚠️ Please tick at least one value to remove, or click 'Skip' to keep all values.");
    return;
  }
  if (!state.target) {
    message(resultDiv, "error", "❌ Error: Target column not set. Please choose it in Explore first.");
    return;
  }
  setContent(resultDiv, "Removing selected values...");
  try {
    const result = await send("/api/v1/dataset/filter", "POST", { target: state.target, remove: selectedValues });
    message(resultDiv, "success", "✅ " + result.message);
    populateTargetValueFilter(result.new_class_counts);
    emit("dataset:changed");
    byId("analyzeClassBtn").click();
  } catch (error) {
    message(resultDiv, "error", "❌ Error: " + error.message);
  }
}

// Balancing preview on the training split (/api/v1/dataset/balancing-preview changes nothing)
async function previewBalancing() {
  const resultDiv = byId("balancingPreviewResult");
  const target = state.target;
  const features = checkedValues(byId("inputFeatures")).filter(f => f !== target);
  setContent(resultDiv, "Running the balancing method on the training split...");
  try {
    const result = await send("/api/v1/dataset/balancing-preview", "POST", {
      target: target, balancing: byId("balancingMethod").value, test_size: testSize(), features: features });
    setContent(resultDiv,
      el("div", { class: "info-message" }, result.message),
      el("p", {}, el("strong", {}, balancingName(result.balancer)),
         ` on ${result.rows.toLocaleString()} training rows`,
         result.sampled && ` (a stratified sample of ${result.train_rows.toLocaleString()})`, ":"),
      el("div", { class: "table-container" },
        table("metrics-table", ["Class", "Before", "After"], Object.keys(result.before).map(cls => [
          cls, result.before[cls].toLocaleString(), (result.after[cls] || 0).toLocaleString()]))),
      !result.changes_rows && el("p", {}, "The number of rows does not change with this method."));
  } catch (error) {
    resultDiv.textContent = "Error: " + error.message;
  }
}

// Feature scores (training split only, after the leakage guard); the best k get ticked
async function recommendFeatures(event) {
  event.preventDefault();
  const resultDiv = byId("recommendationResult");
  const k = byId("kValue").value;
  const target = byId("targetForRec").value;
  if (!k || !target) {
    setContent(resultDiv, "Please provide a valid k value and select a target.");
    return;
  }
  setContent(resultDiv, "Scoring features on the training split...");
  try {
    const result = await send("/api/v1/dataset/feature-scores", "POST",
                              { target: target, k: parseInt(k), test_size: testSize() });
    setContent(resultDiv,
      el("p", {}, `Scored on ${result.rows.toLocaleString()} training rows (${result.method}). The recommended features are now ticked in the feature list.`),
      el("strong", {}, "Recommended Features:"), el("br"),
      result.recommended_features.map(([name, score]) => [`${name} (Score: ${scoreText(score)})`, el("br")]),
      result.recommended_features.some(([, score]) => score === "inf") &&
        el("p", { class: "section-description" }, INFINITE_SCORE_HINT),
      result.blocked.length > 0 && [el("br"), el("strong", {}, "Blocked by the leakage guard:"), el("br"),
        result.blocked.map(item => [`${item.column} (${item.reason})`, el("br")])]);
    setChecked(byId("inputFeatures"), result.recommended_features.map(item => item[0]));
  } catch (error) {
    resultDiv.textContent = "Error: " + error.message;
  }
}

// A new dataset: forget the previous analysis; the profile's label is the first target
function resetForDataset({ columns, dataset }) {
  const label = dataset.profile.label_column;
  ["targetForDist", "targetForRec"].forEach(id => {
    fillSelect(byId(id), columns.map(col => ({ value: col, text: col, selected: col === label })));
  });
  byId("balancingMethod").value = "none";
  setContent(byId("balancingPreviewResult"));
  setContent(byId("recommendationResult"));
  chooseTarget();
}

export function setup() {
  byId("targetForDist").addEventListener("change", chooseTarget);
  byId("analyzeClassBtn").addEventListener("click", analyzeClasses);
  byId("removeTargetValuesBtn").addEventListener("click", removeClasses);
  byId("skipTargetFilterBtn").addEventListener("click", () => {
    message(byId("targetFilterResult"), "info", "ℹ️ Keeping all target values.");
  });
  byId("balancingMethod").addEventListener("change", function() {
    showBalancingHint(this.value, "Select a method to handle class imbalance.");
  });
  byId("previewBalancingBtn").addEventListener("click", previewBalancing);
  byId("recommendBtn").addEventListener("click", recommendFeatures);
  on("dataset:loaded", resetForDataset);
}
