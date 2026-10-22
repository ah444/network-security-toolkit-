// Data: "Do you have a trained model?", dataset upload and preview, bundle load, and
// clearing the workspace.

import { getJSON, send } from "./api.js";
import { byId, el, setContent, show, table } from "./dom.js";
import { fmt } from "./format.js";
import { emit, on } from "./state.js";

// Reset a form's file inputs and their "Choose file..." placeholders
function resetFileInputs(formId) {
  const form = byId(formId);
  form.reset();
  form.querySelectorAll('input[type="file"]').forEach(input => {
    input.value = "";
    const placeholder = input.parentNode.querySelector(".file-input-placeholder");
    if (placeholder) {
      placeholder.textContent = input.multiple ? "Choose file(s)..." : "Choose file...";
    }
  });
}

// Show the chosen file names in place of the placeholder
function watchFileInput(input, placeholder, emptyText) {
  input.addEventListener("change", () => {
    const files = Array.from(input.files);
    placeholder.textContent = files.length === 0 ? emptyText :
      `${files.length} file(s) selected: ${files.map(f => f.name).join(", ")}`;
  });
}

// Show the upload path or the bundle path; the steps start again from Data
function toggleSections() {
  const bundlePath = byId("optionTrained").checked;
  if (bundlePath) {
    resetFileInputs("uploadForm");
    setContent(byId("uploadResult"));
    setContent(byId("dynamicInputs"));
    setContent(byId("classDistributionResult"));
  } else {
    resetFileInputs("loadModelForm");
    setContent(byId("loadModelResult"));
  }
  show(byId("uploadDatasetSection"), !bundlePath);
  show(byId("loadModelSection"), bundlePath);
  ["dataPreviewSection","predictionSection", "exportSection", "loadedFeaturesSection"]
    .forEach(id => show(byId(id), false));
  emit("path:changed", bundlePath ? "bundle" : "upload");
}

function previewTable(columns, rows) {
  return table("preview-table", columns, rows.map(row => columns.map(col => {
    let value = row[col];
    if (typeof value === "string" && value.length > 30) {  // long values are truncated
      value = value.substring(0, 27) + "...";
    }
    return value !== null && value !== undefined ? value : "";
  })));
}

async function fetchDataPreview() {
  const dataStats = byId("dataStats");
  const dataPreviewTable = byId("dataPreviewTable");
  setContent(dataStats, "Loading preview...");
  setContent(dataPreviewTable);
  try {
    const result = await getJSON("/api/v1/dataset/preview",
                                 { rows: parseInt(byId("previewRows").value || 10) });
    const stats = result.stats;
    const stat = (name, value) => el("div", { class: "stat-item" }, el("strong", {}, name), " " + value);
    setContent(dataStats, el("div", { class: "stats-grid" },
      stat("Total Rows:", stats.total_rows.toLocaleString()),
      stat("Total Columns:", stats.total_columns),
      stat("Numeric Columns:", stats.numeric_columns),
      stat("Categorical Columns:", stats.categorical_columns)));
    setContent(dataPreviewTable, previewTable(result.columns, result.preview));
  } catch (error) {
    dataStats.textContent = "Error: " + error.message;
  }
}

// The upload's answer (also after a reload, from GET /api/v1/workspace)
function showDataset(result) {
  const profile = result.dataset.profile;
  setContent(byId("uploadResult"),
    el("p", {}, result.message),
    el("p", {}, "Detected dataset: ", el("strong", {}, profile.name), ` (${profile.variant}); label column: `,
       el("strong", {}, profile.label_column || "not detected"), `; split: ${profile.split_strategy}.`),
    profile.blocked_columns.length > 0 &&
      el("p", {}, "Never used as features (label-derived, IDs or time): " + profile.blocked_columns.join(", ")));
  emit("dataset:loaded", { columns: result.columns, dataset: result.dataset });
  show(byId("dataPreviewSection"));
  fetchDataPreview();
}

async function uploadDataset(event) {
  event.preventDefault();
  const uploadResult = byId("uploadResult");
  const files = byId("dataset").files;
  if (files.length === 0) {
    setContent(uploadResult, "Error: No files selected.");
    return;
  }
  setContent(uploadResult, "Uploading and reading the file(s)...");
  const formData = new FormData();
  Array.from(files).forEach(file => formData.append("dataset", file));
  try {
    showDataset(await send("/api/v1/dataset", "POST", formData));
  } catch (error) {
    uploadResult.textContent = "Error: " + error.message;
  }
}

// The bundle load's answer (also after a reload)
function showBundle(result) {
  setContent(byId("loadModelResult"),
    el("p", {}, result.message),
    el("p", {}, `Trained on ${result.dataset || "unknown data"} at ${result.created_at}.`),
    el("div", { class: "table-container" },
      table("metrics-table", ["Model", "CV macro-F1", "Test macro-F1"], result.models.map(model => [
        model.name + (model.id === result.selected ? " (selected)" : ""),
        fmt(model.cv_macro_f1, 3), fmt(model.test_macro_f1, 3)]))),
    result.warnings.map(warning => el("div", { class: "warning-message" }, warning)));
  emit("bundle:loaded", result);
}

// Only signed bundles from this install are accepted
async function loadBundle(event) {
  event.preventDefault();
  const loadResult = byId("loadModelResult");
  const fileInput = byId("modelFile");
  if (fileInput.files.length === 0) {
    setContent(loadResult, "Error: Please select a model file.");
    return;
  }
  setContent(loadResult, "Checking and loading the bundle...");
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  try {
    showBundle(await send("/api/v1/bundle", "POST", formData));
  } catch (error) {
    loadResult.textContent = "Error: " + error.message;
  }
}

// The server forgets this session's dataset and models
async function clearWorkspace() {
  if (!window.confirm("Remove the uploaded dataset and trained models from this session?")) {
    return;
  }
  try {
    await send("/api/v1/workspace", "DELETE");
    window.location.reload();
  } catch (error) {  // e.g. 409 while a training runs: show the server's message
    byId("clearWorkspaceResult").textContent = "Error: " + error.message;
  }
}

// After a reload: the dataset path when the session has a dataset, else its bundle
function resumeData(answer) {
  if (answer.dataset) {
    showDataset(answer.dataset);
  } else if (answer.bundle) {
    byId("optionTrained").checked = true;
    toggleSections();
    showBundle(answer.bundle);
  }
}

export function setup() {
  byId("optionTrained").addEventListener("change", toggleSections);
  byId("optionNotTrained").addEventListener("change", toggleSections);
  toggleSections();
  watchFileInput(byId("dataset"), byId("dataset").parentNode.querySelector(".file-input-placeholder"),
                 "Choose files...");
  watchFileInput(byId("modelFile"), byId("modelFilePlaceholder"), "Choose file...");
  byId("uploadForm").addEventListener("submit", uploadDataset);
  byId("refreshPreviewBtn").addEventListener("click", fetchDataPreview);
  byId("loadModelForm").addEventListener("submit", loadBundle);
  byId("clearWorkspaceBtn").addEventListener("click", clearWorkspace);
  on("dataset:changed", fetchDataPreview);
  on("workspace:resumed", resumeData);
}
