// Use: prediction with the session's run or a loaded bundle, and exporting the run as a
// signed bundle.

import { send } from "./api.js";
import { byId, el, fillSelect, setContent, show } from "./dom.js";
import { fmt } from "./format.js";
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
  on("run:ready", result => {
    fillSelect(byId("algorithmSelect"), result.leaderboard.map(row => ({
      value: row.model, text: `${row.title} (CV macro-F1 ${fmt(row.cv_macro_f1_mean, 3)})`, selected: row.selected })));
    buildPredictionForm("dynamicInputs", "session-feature-", result.features_info);
    show(byId("predictionSection"));
    show(byId("exportSection"));
    setContent(byId("exportResult"));
  });
  on("bundle:loaded", result => {
    fillSelect(byId("loadedAlgorithmSelect"), result.models.map(model => ({
      value: model.id, text: model.name, selected: model.id === result.selected })));
    buildPredictionForm("loadedDynamicInputs", "loaded-feature-", result.features_info);
    show(byId("loadedFeaturesSection"));
  });
  on("workspace:resumed", answer => {  // after train.js has shown the run
    if (answer.export) {
      showExport(answer.export);
    }
  });
}
