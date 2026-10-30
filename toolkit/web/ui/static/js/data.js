// Data: "Do you have a trained model?", dataset upload and preview, bundle load, and
// clearing the workspace.

import { getJSON, send } from "./api.js";
import { byId, el, scrollBox, setContent, show, table, watchFileInput } from "./dom.js";
import { fmt, wholeNumber } from "./format.js";
import { emit, on } from "./state.js";
import { chips, code, quiet, statTile, summaryPanel } from "./summary.js";

const NO_FILES = "Choose file(s)...";
const UPLOAD_LABEL = "Upload File";
const EXCLUDED_NOTE = "Columns that are label-derived, IDs or time are never used as features.";

// Reset a form's file inputs; their change listeners put back the "Choose file..." texts
function resetFileInputs(formId) {
  const form = byId(formId);
  form.reset();
  form.querySelectorAll('input[type="file"]').forEach(input => {
    input.value = "";
    input.dispatchEvent(new Event("change"));
  });
}

// The dropzone: "N file(s) selected" and a chip per chosen file. The input lies over the
// chips, so it carries the full names as its own tooltip.
function showChosenFiles() {
  const input = byId("dataset");
  const names = Array.from(input.files, file => file.name);
  setContent(byId("datasetChosen"), names.length === 0 ? NO_FILES : `${names.length} file(s) selected`);
  setContent(byId("datasetChips"), chips(names));
  show(byId("datasetChips"), names.length > 0);
  if (names.length > 0) {
    input.title = names.join("\n");
  } else {
    input.removeAttribute("title");
  }
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

// The upload's panel, from the answer's structured fields only (``dataset``, see
// docs/web_api.md); the tile of the official test file only when there is one
export function datasetSummary(dataset) {
  const profile = dataset.profile;
  const excluded = profile.blocked_columns;
  return summaryPanel({
    id: "uploadSummary",
    title: "Uploaded successfully",
    tiles: [
      statTile("Dataset", [code(profile.name), " ", quiet(`(${profile.variant})`)]),
      statTile("Label column", profile.label_column ? code(profile.label_column) : quiet("not detected")),
      statTile("Split", profile.split_strategy),
      dataset.test_file ? statTile("Official test file", code(dataset.test_file),
                                   `${wholeNumber(dataset.test_rows)} rows`) : null,
      statTile("Files processed", wholeNumber(dataset.files.length)),
      statTile("Total rows", wholeNumber(dataset.rows)),
      statTile("Memory used", `${fmt(dataset.memory_mb, 1)} MB`),
      statTile("Excluded from features", excluded.length > 0 ? code(excluded.join(", ")) : quiet("None")),
    ],
    foot: EXCLUDED_NOTE,
  });
}

// A failed upload: the server's message, as text; announced at once (role="alert")
export function uploadError(message) {
  return summaryPanel({ id: "uploadError", kind: "error", title: "Upload failed",
                        body: el("p", {}, message) });
}

// The upload's answer (also after a reload, from GET /api/v1/workspace)
function showDataset(result) {
  setContent(byId("uploadResult"), datasetSummary(result.dataset));
  emit("dataset:loaded", { columns: result.columns, dataset: result.dataset });
  show(byId("dataPreviewSection"));
  fetchDataPreview();
}

// While a file is sent the button is off and says so; the result area is busy (its screen
// reader announcement waits for the answer) and the old result is gone, so results never stack
function uploadButton() {
  return byId("uploadForm").querySelector('button[type="submit"]');
}

function setUploading(busy) {
  const button = uploadButton();
  button.disabled = busy;
  setContent(button, busy ? "Uploading…" : UPLOAD_LABEL);
  [button, byId("uploadResult")].forEach(node =>
    busy ? node.setAttribute("aria-busy", "true") : node.removeAttribute("aria-busy"));
}

let uploading = false;

async function uploadDataset(event) {
  event.preventDefault();
  const uploadResult = byId("uploadResult");
  const files = byId("dataset").files;
  if (uploading) {
    return;  // a second submit (Enter twice) while the first upload runs
  }
  if (files.length === 0) {
    setContent(uploadResult, uploadError("No files selected."));
    return;
  }
  // A disabled button loses the keyboard focus: it gets it back after, if it had it
  const focused = document.activeElement === uploadButton();
  uploading = true;
  setUploading(true);
  setContent(uploadResult);
  const formData = new FormData();
  Array.from(files).forEach(file => formData.append("dataset", file));
  try {
    showDataset(await send("/api/v1/dataset", "POST", formData));
  } catch (error) {
    setContent(uploadResult, uploadError(error.message));
  } finally {
    uploading = false;
    setUploading(false);
    if (focused && document.activeElement === document.body) {
      uploadButton().focus();
    }
  }
}

// The bundle load's answer (also after a reload)
function showBundle(result) {
  setContent(byId("loadModelResult"),
    el("p", {}, result.message),
    el("p", {}, `Trained on ${result.dataset || "unknown data"} at ${result.created_at}.`),
    scrollBox("Models in the bundle",
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
  byId("dataset").addEventListener("change", showChosenFiles);
  watchFileInput(byId("modelFile"), byId("modelFilePlaceholder"), "Choose file...");
  byId("uploadForm").addEventListener("submit", uploadDataset);
  byId("refreshPreviewBtn").addEventListener("click", fetchDataPreview);
  byId("loadModelForm").addEventListener("submit", loadBundle);
  byId("clearWorkspaceBtn").addEventListener("click", clearWorkspace);
  on("dataset:changed", fetchDataPreview);
  on("workspace:resumed", resumeData);
}
