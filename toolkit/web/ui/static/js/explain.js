// What the model relies on (v5.3 F3e), in the Train step: the permutation importance of one of
// the session's models, calculated only when Calculate is clicked. It is a job in the
// training's queue, followed and cancelled like a training; asked again for the same model,
// the server answers the same job, so after a reload Calculate picks it up again.

import { getJSON, send } from "./api.js";
import { importanceView } from "./charts.js";
import { byId, fillSelect, message, setContent, show } from "./dom.js";
import { modelTitle, on } from "./state.js";

let jobUrl = null;
let runId = null;  // the session's run: a result of an earlier one is not shown here

function showJob(job) {
  byId("importanceProgressBar").value = job.percent;
  byId("importanceProgressText").textContent = job.state === "queued" ?
    "Waiting for the training or another calculation to finish" :
    `${job.phase} (${job.percent}%)` + (job.state === "cancelling" ? ". Cancelling..." : "");
}

// Keyboard focus never falls to the page body when Cancel, which it is on, gets disabled or
// hidden: it goes to the result
function endJob() {
  const active = document.activeElement;
  const refocus = !active || active === document.body || byId("importanceProgress").contains(active);
  byId("importanceBtn").disabled = false;
  byId("cancelImportanceBtn").disabled = true;
  show(byId("importanceProgress"), false);
  if (refocus) {
    byId("importanceResult").focus();
  }
}

function showResult(result) {
  const box = byId("importanceResult");
  if (result.run_id !== runId) {
    message(box, "info", "That calculation was for an earlier run; it is on the Saved Runs page.");
  } else {
    setContent(box, importanceView(result, modelTitle(result.model)));
  }
}

async function poll() {
  const box = byId("importanceResult");
  try {
    const job = await getJSON(jobUrl);
    showJob(job);
    if (job.state === "done") {
      showResult(job.result);
      endJob();
    } else if (job.state === "failed" || job.state === "cancelled") {
      message(box, job.state === "failed" ? "error" : "info",
              (job.state === "failed" ? "Error: " : "") + job.error.message);
      endJob();
    } else {
      setTimeout(poll, 1000);
    }
  } catch (error) {  // e.g. the workspace was dropped
    message(box, "error", "Error: " + error.message);
    endJob();
  }
}

async function calculate() {
  const box = byId("importanceResult");
  setContent(box, "Calculating: each feature's values are shuffled in turn on the test rows...");
  byId("importanceBtn").disabled = true;
  try {
    const job = await send("/api/v1/jobs/importance", "POST", { model: byId("importanceModel").value });
    jobUrl = job.url;
    show(byId("importanceProgress"));
    byId("cancelImportanceBtn").disabled = job.state === "done";
    byId("cancelImportanceBtn").focus();
    showJob(job);
    setTimeout(poll, job.state === "done" ? 0 : 1000);
  } catch (error) {
    message(box, "error", "Error: " + error.message);
    endJob();
  }
}

async function cancel() {
  if (!jobUrl) {
    return;
  }
  byId("importanceProgress").focus();
  byId("cancelImportanceBtn").disabled = true;
  try {
    showJob(await send(jobUrl, "DELETE"));
  } catch (error) {
    // the next poll shows how the job ended
  }
}

// A new run (trained, or back after a reload): its models, the selected one first chosen
function runReady(result) {
  runId = result.run_id;
  fillSelect(byId("importanceModel"), result.leaderboard.map(row =>
    ({ value: row.model, text: row.title, selected: row.selected })));
  setContent(byId("importanceResult"));
  show(byId("importanceSection"));
}

export function setup() {
  byId("importanceBtn").addEventListener("click", calculate);
  byId("cancelImportanceBtn").addEventListener("click", cancel);
  on("run:ready", runReady);
}
