// Train: feature, target and model lists, the training job (submit, progress, cancel),
// the results (split, leakage guard, leaderboard) and the confusion matrices.

import { getJSON, send } from "./api.js";
import { leaderboardTable, runCharts } from "./charts.js";
import { byId, checkedValues, el, fillCheckboxList, fillSelect, message, scrollBox, setContent, show } from "./dom.js";
import { balancingName, splitMessage } from "./format.js";
import { emit, modelTitle, on, state, testSize } from "./state.js";

// Fill the target list and the feature list; the profile's label is preselected
function populateColumns({ columns, dataset }) {
  const label = dataset.profile.label_column;
  const blocked = dataset.profile.blocked_columns || [];
  fillSelect(byId("targetColumn"), columns.map(col => ({ value: col, text: col, selected: col === label })));
  fillCheckboxList(byId("inputFeatures"), columns
    .filter(col => col !== label)
    .map(col => ({ value: col, text: col, checked: !blocked.includes(col),
                   title: blocked.includes(col) ? "Blocked by the dataset profile" : "" })));
}

async function loadAvailableModels() {
  const list = byId("modelSelection");
  setContent(list);
  try {
    const result = await getJSON("/api/v1/models");
    result.models.forEach(model => {
      state.modelTitles[model.id] = model.name;
    });
    fillCheckboxList(list, result.models.map(model => ({
      value: model.id, text: model.name, title: model.description, checked: model.default })));
  } catch (error) {
    console.error("Error loading models:", error);
  }
}

// Cell shading: 5 steps of the row share (capped at 50%, so dark text stays >= 4.5:1)
function shade(share) {
  return share <= 0 ? 0 : Math.min(5, Math.ceil(Math.min(share, 0.5) * 10));
}

function shortLabel(label) {
  return typeof label === "string" && label.length > 12 ? label.substring(0, 10) + "..." : label;
}

// A matrix with more classes than this gets smaller cells (v5.2 step 2.2b)
const COMPACT_ABOVE = 15;

function displayConfusionMatrix(matrix, labels, modelName) {
  const container = byId("confusionMatrixDisplay");
  if (!matrix || matrix.length === 0) {
    setContent(container, el("p", {}, "No confusion matrix data available."));
    return;
  }
  const names = labels && labels.length === matrix.length ? labels : matrix.map((_, i) => `Class ${i}`);
  const rows = matrix.map((row, i) => {
    const rowTotal = row.reduce((sum, value) => sum + value, 0);
    return el("tr", {},
      el("th", { scope: "row", title: names[i] }, shortLabel(names[i])),
      row.map((value, j) => {
        const share = rowTotal > 0 ? value / rowTotal : 0;
        const level = shade(share);
        const kind = i === j ? "cm-correct-" : "cm-wrong-";  // diagonal = correct predictions
        return el("td", { class: "cm-cell" + (level ? " " + kind + level : ""),
                          title: `${value} (${(share * 100).toFixed(1)}%)` }, value);
      }));
  });
  const compact = matrix.length > COMPACT_ABOVE ? " cm-compact" : "";
  setContent(container,
    el("div", { class: "cm-header" }, el("strong", {}, modelTitle(modelName)), " - Test Set"),
    // the box scrolls both ways; the header row and the class names stay in view (app.css)
    scrollBox("Confusion matrix of " + modelTitle(modelName),
      el("table", { class: "confusion-matrix" + compact },
        el("thead", {}, el("tr", {}, el("th", { scope: "col" }, "Actual ↓", el("br"), "Predicted →"),
          names.map(name => el("th", { scope: "col", title: name }, shortLabel(name))))),
        el("tbody", {}, rows))),
    el("div", { class: "cm-legend" },
      el("span", { class: "cm-correct" }, "■ Correct"), " ", el("span", { class: "cm-incorrect" }, "■ Incorrect")));
}

function showConfusionMatrix() {
  const modelName = byId("cmModelSelect").value;
  const container = byId("confusionMatrixDisplay");
  if (!modelName) {
    setContent(container, el("p", {}, "Please select a model."));
    return;
  }
  const data = state.confusion[modelName];
  if (data) {
    displayConfusionMatrix(data.matrix, data.labels, modelName);
  } else {
    setContent(container, el("p", {}, "No confusion matrix for this model."));
  }
}

// Training results: split, leakage guard, leaderboard (CV on train + test scored once)
function trainingResult(result) {
  const split = result.split;
  const warning = (text, title) => el("div", { class: "warning-message", title: title }, text);
  return [
    el("p", {}, result.message),
    el("div", { class: "info-message" },
       `Split: ${split.strategy} — ${split.train_rows.toLocaleString()} training rows, ${split.test_rows.toLocaleString()} test rows`,
       split.dedupe && split.dedupe.removed > 0 && `; ${split.dedupe.removed.toLocaleString()} duplicate rows removed before splitting`,
       `. Cross-validation: ${result.cv.folds} folds on ${result.cv.rows.toLocaleString()} training rows${result.cv.sampled ? " (sample)" : ""}.`),
    split.reason_codes.length > 0 &&
      warning(`The ${split.requested} split could not be used: ${split.reason_codes.map(splitMessage).join(" ")}`),
    split.warnings.map(code => warning(splitMessage(code), code)),
    result.leakage.blocked.length > 0 &&
      warning("Left out by the leakage guard: " + result.leakage.blocked.map(b => b.column + " (" + b.reason + ")").join(", ")),
    el("p", {}, `Balancing: ${balancingName(result.balancing.method)}${result.balancing.requested === "recommended" ? " (recommended from the training split)" : ""}.`),
    el("strong", {}, "Leaderboard"), " (ranked by CV macro-F1; ★ = selected):", el("br"),
    leaderboardTable(result.leaderboard),
    runCharts(result.leaderboard, result.folds),
    result.failures.map(failure => warning(`${modelTitle(failure.name)} failed and was left out.`)),
    result.skipped.map(([name, reason]) => warning(`${modelTitle(name)} skipped: ${reason}`)),
    result.classes_only_in_test.length > 0 &&
      warning("Classes only in the test split (no model can predict them): " + result.classes_only_in_test.join(", ")),
  ];
}

function showTrainingResult(result) {
  result.leaderboard.forEach(row => {  // titles also when the run comes back after a reload
    state.modelTitles[row.model] = row.title;
  });
  setContent(byId("configResult"), trainingResult(result));
  state.confusion = result.confusion_matrices;
  fillSelect(byId("cmModelSelect"), result.leaderboard.map(row =>
    ({ value: row.model, text: row.title, selected: row.selected })));
  show(byId("confusionMatrixSection"));
  const chosen = state.confusion[result.selected];
  if (chosen) {
    displayConfusionMatrix(chosen.matrix, chosen.labels, result.selected);
  }
  emit("run:ready", result);
}

// Training runs as a job on the server (one at a time per workspace, 409 otherwise):
// submit it, then ask for its state every second; Cancel stops it at the next step.
let jobUrl = null;
const TRAINING_MESSAGE = "Training models (cross-validation on the training split, then one test)... this can take a while on big datasets.";

function showJob(job) {
  const progressText = byId("trainingProgressText");
  byId("trainingProgressBar").value = job.percent;
  const minutes = Math.floor(job.elapsed_seconds / 60);
  const seconds = Math.round(job.elapsed_seconds % 60);
  if (job.state === "queued") {
    const ahead = job.queue_position;
    progressText.textContent = ahead === 0 ? "Starting..." :
      "Waiting for " + ahead + " other training" + (ahead === 1 ? "" : "s") + " to finish";
  } else {
    progressText.textContent = "Step " + job.done + " of " + job.total + " (" + job.percent + "%): " +
      job.phase + ", " + minutes + " min " + seconds + " s" +
      (job.state === "cancelling" ? ". Cancelling after the current model..." : "");
  }
}

// Keyboard focus never falls to the page body when a button it is on gets disabled or
// hidden: Train → Cancel when the job starts, Cancel → the progress, then the result.
function focusLost(region) {
  const active = document.activeElement;
  return !active || active === document.body || region.contains(active);
}

function endJob() {
  const refocus = focusLost(byId("trainingProgress"));
  byId("configForm").querySelector('button[type="submit"]').disabled = false;
  byId("cancelTrainingBtn").disabled = true;
  show(byId("trainingProgress"), false);
  if (refocus) {
    byId("configResult").focus();
  }
}

async function poll() {
  const configResult = byId("configResult");
  try {
    const job = await getJSON(jobUrl);
    showJob(job);
    if (job.state === "done") {
      showTrainingResult(job.result);
      endJob();
    } else if (job.state === "failed" || job.state === "cancelled") {
      configResult.textContent = (job.state === "failed" ? "Error: " : "") + job.error.message;
      endJob();
    } else {
      setTimeout(poll, 1000);
    }
  } catch (error) {  // e.g. the workspace was dropped
    configResult.textContent = "Error: " + error.message;
    endJob();
  }
}

// Follow a job that was still queued or running when the page was reloaded
function followJob(job) {
  jobUrl = job.url;
  setContent(byId("configResult"), TRAINING_MESSAGE);
  byId("configForm").querySelector('button[type="submit"]').disabled = true;
  show(byId("trainingProgress"));
  byId("cancelTrainingBtn").disabled = job.state === "cancelling";
  showJob(job);
  setTimeout(poll, 1000);
}

async function cancelJob() {
  if (!jobUrl) {
    return;
  }
  byId("trainingProgress").focus();
  byId("cancelTrainingBtn").disabled = true;
  try {
    showJob(await send(jobUrl, "DELETE"));
  } catch (error) {
    // the next poll shows how the job ended
  }
}

async function train(event) {
  event.preventDefault();
  const configResult = byId("configResult");
  const target = byId("targetColumn").value;
  const features = checkedValues(byId("inputFeatures")).filter(f => f !== target);
  const models = checkedValues(byId("modelSelection"));
  if (features.length === 0) {
    message(configResult, "error", "❌ Please tick at least one input feature.");
    return;
  }
  if (models.length === 0) {
    message(configResult, "error", "❌ Please tick at least one model to train.");
    return;
  }
  setContent(configResult, TRAINING_MESSAGE);
  configResult.focus();
  byId("configForm").querySelector('button[type="submit"]').disabled = true;
  jobUrl = null;
  try {
    const job = await send("/api/v1/jobs/training", "POST", {
      features: features, target: target, balancing: byId("balancingMethod").value,
      test_size: testSize(), models: models });
    jobUrl = job.url;
    show(byId("trainingProgress"));
    byId("cancelTrainingBtn").disabled = false;
    byId("cancelTrainingBtn").focus();
    showJob(job);
    setTimeout(poll, 1000);
  } catch (error) {
    configResult.textContent = "Error: " + error.message;
    endJob();
  }
}

// The target chosen in Explore, pre-filled here (still changeable)
function prefillTarget(target) {
  if (target) {
    byId("targetColumn").value = target;
    untickTarget();
  }
}

// The target can't also be an input feature
function untickTarget() {
  const target = byId("targetColumn").value;
  byId("inputFeatures").querySelectorAll("input[type=checkbox]").forEach(box => {
    if (box.value === target) {
      box.checked = false;
    }
  });
}

export function setup() {
  byId("targetColumn").addEventListener("change", untickTarget);
  // "Select all" / "Clear" buttons of a checkbox list
  document.querySelectorAll("button[data-list]").forEach(button => {
    button.addEventListener("click", () => {
      const target = byId("targetColumn").value;
      byId(button.dataset.list).querySelectorAll("input[type=checkbox]").forEach(box => {
        box.checked = button.dataset.check === "all" && box.value !== target;
      });
    });
  });
  byId("trainTestSlider").addEventListener("input", function() {
    byId("splitLabel").textContent = `Train: ${100 - parseInt(this.value)}% | Test: ${parseInt(this.value)}%`;
  });
  byId("configForm").addEventListener("submit", train);
  byId("cancelTrainingBtn").addEventListener("click", cancelJob);
  byId("showConfusionMatrixBtn").addEventListener("click", showConfusionMatrix);
  on("dataset:loaded", detail => {
    populateColumns(detail);
    loadAvailableModels();
  });
  on("target:chosen", prefillTarget);
  on("workspace:resumed", answer => {
    if (answer.run) {
      showTrainingResult(answer.run);
    }
    if (answer.job) {
      followJob(answer.job);
    }
  });
}
