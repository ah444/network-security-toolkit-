// Use: prediction with the session's run or a loaded bundle (one row, or every row of a
// CSV file), and exporting the run as a signed bundle.

import { send } from "./api.js";
import { byId, el, fillSelect, setContent, show, table, watchFileInput } from "./dom.js";
import { on } from "./state.js";

// One input per feature: a number box, a list of the known categories, or a text box
function buildPredictionForm(containerId, idPrefix, featuresInfo) {
  setContent(byId(containerId), featuresInfo.map((feature, index) => {
    let input;
    if (feature.type === "numeric") {
      input = el("input", { type: "number", step: "any" });
    } else if (feature.type === "categorical") {
      input = el("select", {}, feature.classes.map(cls => el("option", { value: cls, text: cls })));
    } else {
      input = el("input", { type: "text" });
    }
    input.id = idPrefix + index;
    input.name = feature.name;
    input.required = true;
    return el("div", { class: "form-group" }, el("label", { htmlFor: input.id }, feature.name + ":"), input);
  }));
}

// Values of a prediction form: feature name -> entered text
function formValues(containerId) {
  const values = {};
  byId(containerId).querySelectorAll("input, select").forEach(input => {
    values[input.name] = input.value;
  });
  return values;
}

function predictWith(url, inputsId, algorithmId, resultId) {
  return async event => {
    event.preventDefault();
    const resultDiv = byId(resultId);
    resultDiv.textContent = "Predicting...";
    try {
      const result = await send(url, "POST", { values: formValues(inputsId), model: byId(algorithmId).value });
      resultDiv.textContent = "Predicted Value: " + result.prediction;
    } catch (error) {
      resultDiv.textContent = "Error: " + error.message;
    }
  };
}

// The parts of each "Predict a CSV file" form: the session's run or the loaded bundle
const BATCH = {
  run: { url: "/api/v1/run/predict-file", input: "predictFile", algorithm: "algorithmSelect",
         status: "predictFileResult", output: "predictFileOutput", form: "predictFileForm" },
  bundle: { url: "/api/v1/bundle/predict-file", input: "loadedPredictFile",
            algorithm: "loadedAlgorithmSelect", status: "loadedPredictFileResult",
            output: "loadedPredictFileOutput", form: "loadedPredictFileForm" },
};

const count = number => number.toLocaleString("en-US");
const percent = share => (share * 100).toFixed(1) + "%";

// Rows per predicted class, largest first, each with a bar: a <meter>, so no inline style
// is needed (S10); the numbers are in the table, the bar is only a picture of them.
function batchTable(counts) {
  return el("div", { class: "table-container" }, table("batch-table", ["Class", "Rows", "Share"],
    counts.map(row => [row.label, count(row.count),
      el("span", { class: "batch-share" }, percent(row.share),
         el("meter", { class: "batch-bar", min: 0, max: 1, value: row.share, "aria-hidden": "true" }))])));
}

// A batch answer (also after a reload): the status line is announced, the table is not
function showBatch(parts, result) {
  const lines = [result.message];
  if (result.matches) {
    lines.push(`${count(result.matches.matching)} of ${count(result.matches.rows)} predictions ` +
               `(${percent(result.matches.share)}) match the file's ${result.label_column} column.`);
  }
  setContent(byId(parts.status), lines.join(" "));
  setContent(byId(parts.output), batchTable(result.counts),
    el("a", { href: result.download_url, download: result.download_name, class: "btn btn-sm" },
       "Download " + result.download_name));
}

function predictFile(parts) {
  return async event => {
    event.preventDefault();
    const file = byId(parts.input).files[0];
    const body = new FormData();
    body.append("file", file);
    body.append("model", byId(parts.algorithm).value);
    setContent(byId(parts.output));
    byId(parts.status).textContent = "Predicting every row of " + file.name + "...";
    try {
      showBatch(parts, await send(parts.url, "POST", body));
    } catch (error) {
      byId(parts.status).textContent = "Error: " + error.message;
    }
  };
}

function clearBatch(parts) {
  setContent(byId(parts.status));
  setContent(byId(parts.output));
}

// The export's answer, with the download link (also after a reload)
function showExport(result) {
  setContent(byId("exportResult"),
    el("p", {}, result.message),
    el("p", {}, "Run " + result.run_id + ", SHA-256 " + result.sha256.substring(0, 16) + "…"),
    el("a", { href: result.download_url, download: result.download_name, class: "btn btn-sm" },
       "Download " + result.download_name));
}

// A signed bundle, then a download link
async function exportRun(event) {
  event.preventDefault();
  const exportResult = byId("exportResult");
  exportResult.textContent = "Saving the signed bundle...";
  try {
    showExport(await send("/api/v1/run/export", "POST", { name: byId("modelName").value }));
  } catch (error) {
    exportResult.textContent = "Error: " + error.message;
  }
}

export function setup() {
  byId("predictionForm").addEventListener("submit",
    predictWith("/api/v1/run/predict", "dynamicInputs", "algorithmSelect", "predictionResult"));
  byId("predictLoadedForm").addEventListener("submit",
    predictWith("/api/v1/bundle/predict", "loadedDynamicInputs", "loadedAlgorithmSelect", "loadedPredictionResult"));
  byId("exportForm").addEventListener("submit", exportRun);
  for (const parts of Object.values(BATCH)) {
    byId(parts.form).addEventListener("submit", predictFile(parts));
    const input = byId(parts.input);
    watchFileInput(input, input.parentNode.querySelector(".file-input-placeholder"), "Choose file...");
  }
  on("run:ready", result => {
    fillSelect(byId("algorithmSelect"), result.leaderboard.map(row => ({
      value: row.model, text: row.title, selected: row.selected })));
    buildPredictionForm("dynamicInputs", "session-feature-", result.features_info);
    show(byId("predictionSection"));
    show(byId("exportSection"));
    setContent(byId("exportResult"));
    clearBatch(BATCH.run);  // a new run: the last file's labels came from the old one
  });
  on("bundle:loaded", result => {
    fillSelect(byId("loadedAlgorithmSelect"), result.models.map(model => ({
      value: model.id, text: model.name, selected: model.id === result.selected })));
    buildPredictionForm("loadedDynamicInputs", "loaded-feature-", result.features_info);
    show(byId("loadedFeaturesSection"));
    clearBatch(BATCH.bundle);
  });
  on("workspace:resumed", answer => {  // after train.js has shown the run
    if (answer.export) {
      showExport(answer.export);
    }
    if (answer.predictions && BATCH[answer.predictions.source]) {
      showBatch(BATCH[answer.predictions.source], answer.predictions);
    }
  });
}
