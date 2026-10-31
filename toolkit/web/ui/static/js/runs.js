// The saved runs' page (/runs, v5.3 F2), loaded as an ES module: the runs exported from the
// trainer, 50 at a time and newest first (GET /api/v1/runs), one of them with its charts, or
// two compared. It reads the runs' records only: nothing is uploaded or trained, and no model
// is loaded. A run's id after the # (the trainer's export links to it) shows that run.

import { getJSON } from "./api.js";
import { byId, message, setContent, show } from "./dom.js";
import { RUN_ID, comparisonView, runRow, runView } from "./runsview.js";

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
      message(byId("runsStatus"), "info", "No saved runs yet: train a model on the Dynamic IDS page and export it; each export is saved here.");
    } else {
      listedText = `${listed} run${listed === 1 ? "" : "s"} listed${answer.more ? ", older ones below" : ""}.`;
      setContent(byId("runsStatus"), listedText);
    }
  } catch (error) {
    message(byId("runsStatus"), "error", "Error: " + error.message);
  }
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
  });
  byId("runsOlder").addEventListener("click", load);
  byId("compareRuns").addEventListener("click", compare);
  await load();
  choiceChanged();
  showFromHash();
}

setup();
