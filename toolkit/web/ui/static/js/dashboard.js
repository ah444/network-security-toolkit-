// The home page's dashboard (v5.4 F10): GET /api/v1/dashboard in two parts side by side,
// never linked: the scans and their risks, and the models with this session. Every value goes
// in as text (dom.js). See also: web/services/dashboard.py (what each part holds).

import { getJSON } from "./api.js";
import { byId, el, message, setContent } from "./dom.js";
import { fmt } from "./format.js";
import { when } from "./runsview.js";

// a risk's colour, from the scan page's levels; NOT CHECKED is never shown as clean
const RISK_CLASS = { CRITICAL: "cvss-critical", HIGH: "cvss-high", MEDIUM: "cvss-medium",
                     LOW: "cvss-low", "NO KNOWN CVE": "cvss-none" };

const riskClass = risk => RISK_CLASS[risk] || "cvss-unknown";
const plural = (count, word) => `${count} ${word}${count === 1 ? "" : "s"}`;

function nothingYet(text, href, label) {
  return el("p", {}, text + " ", el("a", { href }, label));
}

// One bar per value, its name before and its number after: a <meter>, so no inline style
// (S10); the bar is only a picture of the number beside it
function bars(rows, max) {
  return el("div", { class: "dash-bars" },
    ...rows.flatMap(([name, value, shown]) => [
      el("span", {}, name),
      el("meter", { class: "dash-bar", min: 0, max, value, "aria-hidden": "true" }),
      el("span", {}, shown)]));
}

function target(row) {
  const risk = row.risk + (row.unconfirmed ? " (unconfirmed)" : "");
  return el("li", { class: "dash-item " + riskClass(row.risk) },
    el("strong", {}, row.target), " ", risk,
    el("div", { class: "dash-meta" },
       `${plural(row.open_ports, "open port")}, scanned ${when(row.started)} · `,
       el("a", { href: "/scans/" + encodeURIComponent(row.scan_id) + "/report", target: "_blank",
                 rel: "noopener" }, "its report")));
}

function scansPart(part) {
  if (part.error) {
    return el("p", { class: "status-error" }, part.error);
  }
  if (!part.targets.length) {
    return nothingYet("No saved scans yet.", "/scan", "Start a scan");
  }
  const most = Math.max(...part.by_risk.map(row => row.count));
  return el("div", {},
    el("p", { class: "field-hint" }, `The newest scan of each target, among the ${part.read} newest scans` +
       (part.left_out ? `; ${part.left_out} file(s) that aren't scans left out.` : ".")),
    bars(part.by_risk.map(row => [row.risk, row.count, plural(row.count, "target")]), most),
    el("h3", {}, "Targets, the riskiest first"),
    el("ul", { class: "dash-list" }, ...part.targets.map(target)),
    el("h3", {}, "The newest scans"),
    el("ul", { class: "dash-list" }, ...part.recent.map(scan => el("li", { class: "dash-item" },
      el("strong", {}, scan.target), ` ${when(scan.started)}, ${plural(scan.open_ports, "open port")}`))),
    el("p", {}, el("a", { href: "/scans" }, "All saved scans")));
}

function runsPart(part) {
  if (part.error) {
    return el("p", { class: "status-error" }, part.error);
  }
  if (!part.recent.length) {
    return nothingYet("No saved runs yet.", "/dynamic_ids", "Start training");
  }
  const scored = part.recent.filter(run => run.test_macro_f1 !== null);
  return el("div", {},
    scored.length ? bars(scored.map(run => [when(run.created_at), run.test_macro_f1,
                                            "test macro-F1 " + fmt(run.test_macro_f1, 3)]), 1)
                  : "",
    el("ul", { class: "dash-list" }, ...part.recent.map(run => el("li", { class: "dash-item" },
      el("strong", {}, run.selected_title || "no model selected"), ` on ${run.dataset.name || "a dataset"}`,
      el("div", { class: "dash-meta" }, `${when(run.created_at)} · run ${run.run_id}`)))),
    el("p", {}, el("a", { href: "/runs" }, "All saved runs")));
}

function sessionLines(session) {
  const lines = [];
  if (session.dataset) {
    lines.push(`Dataset: ${session.dataset.files.join(", ")} (${session.dataset.rows.toLocaleString("en-US")} rows)`);
  }
  if (session.training) {
    lines.push(`A training is ${session.training.state}: ${session.training.phase}`);
  }
  if (session.run) {
    lines.push(`Last training: run ${session.run.run_id}` + (session.run.saved ? "" : " (not saved)"));
  }
  if (session.bundle) {
    lines.push(`Loaded bundle: run ${session.bundle.run_id}, ${plural(session.bundle.models, "model")}`);
  }
  if (session.predictions) {
    const found = session.predictions;
    lines.push(`Last predictions (kept in this session only): ${found.file}, ${found.rows} rows`);
  }
  if (session.scan) {
    lines.push(`A scan is ${session.scan.state}: ${session.scan.phase}`);
  }
  if (session.last_scan) {
    lines.push(`Last scan: ${session.last_scan.target}, ${plural(session.last_scan.open_ports, "open port")}`);
  }
  return lines;
}

function sessionPart(session) {
  const lines = sessionLines(session);
  return lines.length ? el("ul", {}, ...lines.map(line => el("li", {}, line)))
    : el("p", { class: "field-hint" }, "Nothing in this session yet: the dataset, the models and the predictions of a training show here.");
}

async function refresh() {
  message(byId("dashboardStatus"), "info", "Reading...");
  try {
    const answer = await getJSON("/api/v1/dashboard");
    setContent(byId("scansPart"), scansPart(answer.scans));
    setContent(byId("runsPart"), runsPart(answer.runs));
    setContent(byId("sessionPart"), sessionPart(answer.session));
    setContent(byId("dashboardStatus"));
  } catch (error) {
    message(byId("dashboardStatus"), "error", "Error: " + error.message);
  }
}

byId("dashboardRefresh").addEventListener("click", refresh);
refresh();
