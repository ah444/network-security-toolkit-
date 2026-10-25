// A running scan (v5.2 step 2.4e): its progress and phase, its log (only the new lines are
// asked for, GET /api/v1/jobs/<id>?since=N) and Cancel. The log's lines are text from the
// server, the network's parts already cleaned there; they go in as text only (dom.js).

import { getJSON, send } from "./api.js";
import { byId, el, message, setContent, show } from "./dom.js";

let jobUrl = null;
let logSize = 0;
let ended = null;  // what to call when the scan ends (follow's onEnd)

// Start pressed: the form waits for the server's answer
export function starting() {
  byId("scanStart").disabled = true;
  setContent(byId("scanStatus"), "Checking the scan...");
}

function showJob(job) {
  byId("scanProgressBar").value = job.percent;
  const minutes = Math.floor(job.elapsed_seconds / 60);
  const seconds = Math.round(job.elapsed_seconds % 60);
  let text;
  if (job.state === "queued") {
    const ahead = job.queue_position;
    text = ahead === 0 ? "Starting..." :
      "Waiting for " + ahead + " other scan" + (ahead === 1 ? "" : "s") + " to finish";
  } else {
    text = job.percent + "%: " + job.phase + ", " + minutes + " min " + seconds + " s" +
      (job.state === "cancelling" ? ". Cancelling..." : "");
  }
  byId("scanProgressText").textContent = text;
  addLog(job.log, job.log_size);
}

// New log lines at the end; the box follows them unless it was scrolled up to read
function addLog(lines, size) {
  const log = byId("scanLog");
  const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
  log.append(...lines.map(line => el("li", {}, line)));
  logSize = size;
  if (atEnd) {
    log.scrollTop = log.scrollHeight;
  }
}

// Keyboard focus never falls to the page body when the button it is on is disabled or
// hidden: Start → Cancel while it runs, Cancel → the progress, then the results or the status
function focusLost(region) {
  const active = document.activeElement;
  return !active || active === document.body || region.contains(active);
}

function end(job) {
  const refocus = focusLost(byId("scanProgress"));
  byId("scanStart").disabled = false;
  byId("scanCancel").disabled = true;
  show(byId("scanProgress"), false);
  jobUrl = null;
  const status = byId("scanStatus");
  if (job !== null && job.state === "done") {
    message(status, "success", "Scan finished: " + job.result.message);
    ended(job.result, refocus);
    return;
  }
  if (job !== null) {  // else its end is unknown, and the error is shown
    message(status, job.state === "failed" ? "error" : "info",
            (job.state === "failed" ? "Error: " : "") + job.error.message);
  }
  if (refocus) {
    status.focus();
  }
  ended(null, false);
}

async function poll() {
  try {
    const job = await getJSON(jobUrl, { since: logSize });
    showJob(job);
    if (["done", "failed", "cancelled"].includes(job.state)) {
      end(job);
    } else {
      setTimeout(poll, 1000);
    }
  } catch (error) {  // e.g. the server was restarted
    message(byId("scanStatus"), "error", "Error: " + error.message);
    end(null);
  }
}

// Follow a scan: just started (``fresh``: Cancel gets the focus), or found running after a
// reload. ``onEnd(answer, focus)`` gets the answer of a finished scan, or null when it was
// cancelled or failed (nothing was kept from it)
export function follow(job, onEnd, fresh) {
  jobUrl = job.url;
  ended = onEnd;
  logSize = 0;
  setContent(byId("scanLog"));
  show(byId("scanLogSection"));
  setContent(byId("scanStatus"), "Scan started. Its steps are in the scan log below.");
  byId("scanStart").disabled = true;
  show(byId("scanProgress"));
  byId("scanCancel").disabled = job.state === "cancelling";
  showJob(job);
  if (fresh) {
    byId("scanCancel").focus();
  }
  setTimeout(poll, 1000);
}

async function cancel() {
  if (!jobUrl) {
    return;
  }
  byId("scanProgress").focus();
  byId("scanCancel").disabled = true;
  try {
    showJob(await send(jobUrl, "DELETE"));
  } catch (error) {
    // the next poll shows how the scan ended
  }
}

export function setup() {
  byId("scanCancel").addEventListener("click", cancel);
}
