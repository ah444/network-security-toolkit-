// The home page's "Other computers" section (HS1, RD1), loaded as an ES module only on this
// computer while the toolkit is open to the network: a new access token locks every other
// computer out at once (the token is printed in the terminal, never sent to the page), and the
// jobs other computers asked for are listed, each with Stop. Values from the server are set
// as text only (dom.js).

import { getJSON, send } from "./api.js";
import { byId, el, message, setContent, show } from "./dom.js";

const KINDS = { training: "Training", importance: "Importance", scan: "Port scan" };

function elapsed(seconds) {
  const whole = Math.round(seconds);
  return whole < 60 ? whole + " s" : Math.floor(whole / 60) + " min " + (whole % 60) + " s";
}

function jobRow(job) {
  const kind = KINDS[job.kind] || job.kind;
  const stopping = job.state === "cancelling";
  const stop = el("button", { type: "button", class: "btn btn-sm btn-danger", disabled: stopping,
                              "aria-label": "Stop the " + kind.toLowerCase() + " of " + job.source },
                  stopping ? "Stopping..." : "Stop");
  stop.addEventListener("click", () => stopJob(job, stop));
  return el("tr", {},
    el("td", {}, el("code", {}, job.source)),
    el("td", {}, kind),
    el("td", {}, stopping ? "Stopping after the current step" : job.phase),
    el("td", {}, elapsed(job.elapsed_seconds)),
    el("td", {}, stop));
}

async function showJobs() {
  try {
    const answer = await getJSON("/api/v1/access/jobs");
    setContent(byId("otherJobsRows"), answer.jobs.map(jobRow));
    show(byId("otherJobsBox"), answer.jobs.length > 0);
    show(byId("noOtherJobs"), answer.jobs.length === 0);
  } catch (error) {
    message(byId("accessStatus"), "error", "Error: " + error.message);
  }
}

async function stopJob(job, button) {
  button.disabled = true;
  try {
    await send(job.url, "DELETE");
  } catch (error) {
    message(byId("accessStatus"), "error", "Error: " + error.message);
  }
  await showJobs();
}

async function newToken() {
  if (!window.confirm("Make a new access token? Every other computer is locked out now: the "
                      + "jobs they asked for that still wait are cancelled, a running one goes on.")) {
    return;
  }
  const button = byId("newToken");
  button.disabled = true;
  try {
    const answer = await send("/api/v1/access/token", "POST");
    message(byId("accessStatus"), "success", answer.message);
  } catch (error) {
    message(byId("accessStatus"), "error", "Error: " + error.message);
  }
  button.disabled = false;
  await showJobs();
}

byId("newToken").addEventListener("click", newToken);
byId("refreshJobs").addEventListener("click", showJobs);
showJobs();
