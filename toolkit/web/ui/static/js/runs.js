// The saved runs' page (/runs, v5.3 F2), loaded as an ES module: the runs exported from the
// trainer, 50 at a time and newest first (GET /api/v1/runs), one of them with its charts, or
// two compared. It reads the runs' records only: nothing is uploaded or trained, and no model
// is loaded. A run's id after the # (the trainer's export links to it) shows that run. Its one
// change: a run is deleted when the user asks, after the page named its files (HR4a). The
// Storage section only shows; Register again adds a kept bundle's record back (HR4b).

import { getJSON, send } from "./api.js";
import { byId, message, setContent, show } from "./dom.js";
import { RUN_ID, comparisonView, deleteView, runRow, runView, storageView } from "./runsview.js";

let last = null;  // the oldest run listed: "Older runs" asks for the ones before it
let leftOut = 0;
let listedText = "";  // what the status says of the list, again after a run is shown

function chosen() {
  return Array.from(byId("runsRows").querySelectorAll("input[type=checkbox]:checked"))
    .map(box => box.value);
}

// Compare needs exactly two runs
function choiceChanged() {
  const count = chosen().length;
  byId("compareRuns").disabled = count !== 2;
  byId("compareHint").textContent = count === 2 ? "Two runs chosen." :
    `Tick two runs to compare them (${count} ticked).`;
}

async function load() {
  try {
    const answer = await getJSON("/api/v1/runs", last ? { before: last } : undefined);
    byId("runsRows").append(...answer.runs.map(runRow));
    if (answer.runs.length > 0) {
      last = answer.runs[answer.runs.length - 1].run_id;
    }
    leftOut += answer.left_out;
    const listed = byId("runsRows").children.length;
    show(byId("runsBox"), listed > 0);
    show(byId("runsOlder"), answer.more && answer.runs.length > 0);
    show(byId("compareControls"), listed > 1);
    byId("runsLeftOut").textContent = `${leftOut} file${leftOut === 1 ? " is" : "s are"} in the runs folder but not a run this toolkit can show (edited, or from a newer version): left out.`;
    show(byId("runsLeftOut"), leftOut > 0);
    if (listed === 0) {
      message(byId("runsStatus"), "info", "No saved runs yet: train a model on the Dynamic IDS page; each training is saved here.");
    } else {
      listedText = `${listed} run${listed === 1 ? "" : "s"} listed${answer.more ? ", older ones below" : ""}.`;
      setContent(byId("runsStatus"), listedText);
    }
  } catch (error) {
    message(byId("runsStatus"), "error", "Error: " + error.message);
  }
}

// The list from the newest run again (after a run came back)
async function reloadRuns() {
  byId("runsRows").replaceChildren();
  last = null;
  leftOut = 0;
  await load();
  choiceChanged();
}

async function loadStorage() {
  try {
    setContent(byId("storageView"), storageView(await getJSON("/api/v1/storage")));
  } catch (error) {
    message(byId("storageStatus"), "error", "Error: " + error.message);
  }
}

async function register(runId) {
  message(byId("storageStatus"), "info", "Checking the bundle of run " + runId + "...");
  try {
    await send("/api/v1/bundles/" + encodeURIComponent(runId) + "/register", "POST");
    message(byId("storageStatus"), "success", `Run ${runId} is a saved run again.`);
    await reloadRuns();
    await loadStorage();
  } catch (error) {
    message(byId("storageStatus"), "error", "Error: " + error.message);
  }
  byId("storageStatus").focus();
}

// Only one of the two views shows at a time; the keyboard focus goes to its heading
function showView(section, heading, title, content) {
  show(byId(section === "runSection" ? "comparisonSection" : "runSection"), false);
  byId(heading).textContent = title;
  setContent(byId(section === "runSection" ? "runView" : "comparisonView"), content);
  show(byId(section));
  byId(heading).focus();
}

async function showRun(runId) {
  message(byId("runsStatus"), "info", "Reading run " + runId + "...");
  try {
    const view = await getJSON("/api/v1/runs/" + encodeURIComponent(runId));
    setContent(byId("runsStatus"), listedText);
    showView("runSection", "runTitle", "Run " + runId, runView(view));
  } catch (error) {
    message(byId("runsStatus"), "error", "Error: " + error.message);
  }
}

// A is the older of the two (a run id starts with its time), so Δ = B − A is newer − older
async function compare() {
  const [a, b] = chosen().sort();
  message(byId("runsStatus"), "info", "Comparing the two runs...");
  try {
    const answer = await getJSON("/api/v1/runs/compare", { a: a, b: b });
    setContent(byId("runsStatus"), listedText);
    showView("comparisonSection", "comparisonTitle", "A and B compared", comparisonView(answer));
  } catch (error) {
    message(byId("runsStatus"), "error", "Error: " + error.message);
  }
}

let deleting = null;  // the run whose files the delete section names

// Delete: first its files, then the user's yes; nothing is removed before that
async function askDelete(runId) {
  message(byId("runsStatus"), "info", "Reading the files of run " + runId + "...");
  try {
    const answer = await getJSON("/api/v1/runs/" + encodeURIComponent(runId) + "/files");
    setContent(byId("runsStatus"), listedText);
    deleting = runId;
    byId("deleteTitle").textContent = "Delete run " + runId + "?";
    setContent(byId("deleteView"), deleteView(answer));
    show(byId("deleteSection"));
    byId("deleteTitle").focus();
  } catch (error) {
    message(byId("runsStatus"), "error", "Error: " + error.message);
  }
}

function closeDelete() {
  show(byId("deleteSection"), false);
  deleting = null;
}

// The list without a deleted run's line, and neither view (it may have shown that run)
function dropRow(runId) {
  byId("runsRows").querySelector(`button[data-delete="${runId}"]`).closest("tr").remove();
  show(byId("runSection"), false);
  show(byId("comparisonSection"), false);
  const listed = byId("runsRows").children.length;
  show(byId("runsBox"), listed > 0);
  show(byId("compareControls"), listed > 1);
  choiceChanged();
  listedText = `${listed} run${listed === 1 ? "" : "s"} listed.`;
}

async function confirmDelete() {
  const runId = deleting;
  const box = byId("deleteView").querySelector("input[name=with_bundle]");
  closeDelete();
  try {
    const answer = await send("/api/v1/runs/" + encodeURIComponent(runId) + "/delete", "POST",
                              { with_bundle: box !== null && box.checked });
    dropRow(runId);
    const count = answer.removed.length;
    message(byId("runsStatus"), "success", `Run ${runId} deleted (${count} file${count === 1 ? "" : "s"} removed)` +
      (answer.kept ? `; its bundle ${answer.kept.name} is kept in instance/bundles.` : "."));
    loadStorage();
  } catch (error) {
    message(byId("runsStatus"), "error", "Error: " + error.message);
  }
  byId("runsStatus").focus();
}

function showFromHash() {
  const runId = window.location.hash.slice(1);
  if (RUN_ID.test(runId)) {
    showRun(runId);
  }
}

async function setup() {
  byId("runsRows").addEventListener("change", choiceChanged);
  byId("runsRows").addEventListener("click", event => {
    const button = event.target.closest("button[data-run]");
    if (button) {
      window.location.replace("#" + button.dataset.run);  // the address names the run shown
      showRun(button.dataset.run);
    }
    const remove = event.target.closest("button[data-delete]");
    if (remove) {
      askDelete(remove.dataset.delete);
    }
  });
  byId("storageView").addEventListener("click", event => {
    const button = event.target.closest("button[data-register]");
    if (button) {
      register(button.dataset.register);
    }
  });
  byId("deleteConfirm").addEventListener("click", confirmDelete);
  byId("deleteCancel").addEventListener("click", closeDelete);
  byId("runsOlder").addEventListener("click", load);
  byId("compareRuns").addEventListener("click", compare);
  await load();
  choiceChanged();
  showFromHash();
  loadStorage();
}

setup();
