// The saved scans' page (/scans, v5.4 F6b), loaded as an ES module: every finished scan, 50 at a
// time and newest first (GET /api/v1/scans), one of them shown with the Port Scanner page's own
// cards (scanresults.js), its report, and Scan again, which opens the Port Scanner with its
// choices (the authorization asked again there), or two compared (F6c, scancompare.js). It
// reads the scan history only: nothing is scanned here. A scan's id after the # (the Port Scanner links to it) shows that scan. Its
// one change: a scan is deleted when the user asks, after the page named it.

import { getJSON, send } from "./api.js";
import { byId, el, message, setContent, show } from "./dom.js";
import { code, quiet } from "./summary.js";
import { comparisonView } from "./scancompare.js";
import { hideResults, showResults } from "./scanresults.js";

const SCAN_ID = /^\d{8}T\d{6}Z-[0-9a-f]{8}$/;
const ENGINES = { builtin: "Built-in", nmap: "nmap" };

let last = null;  // the oldest scan listed: "Older scans" asks for the ones before it
let leftOut = 0;
let listedText = "";  // what the status says of the list, again after a scan is shown
let shown = null;  // the scan shown, if any
let deleting = null;  // the scan the delete section names

function when(started) {
  const match = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})/.exec(started);
  return match ? `${match[1]} ${match[2]} UTC` : started;
}

function hostsText(hosts) {
  return ["scanned", "skipped", "failed"].filter(state => hosts[state] > 0)
    .map(state => `${hosts[state]} ${state}`).join(", ") || "none";
}

// The worst severity of the ports looked up; "no lookup" when no software was recognised
function worstText(scan) {
  if (scan.worst === null) {
    return scan.open_ports > 0 ? "no lookup" : "–";
  }
  return scan.worst + (scan.worst_unconfirmed ? " (unconfirmed)" : "");
}

function chosen() {
  return Array.from(byId("scansRows").querySelectorAll("input[type=checkbox]:checked"))
    .map(box => box.value);
}

// Compare needs exactly two scans
function choiceChanged() {
  const count = chosen().length;
  byId("compareScans").disabled = count !== 2;
  byId("compareHint").textContent = count === 2 ? "Two scans chosen." :
    `Tick two scans to compare them (${count} ticked).`;
}

function scanRow(scan) {
  return el("tr", {},
    el("td", {}, el("label", { class: "choice run-pick" },
      el("input", { type: "checkbox", value: scan.scan_id, "aria-label": "Compare scan " + scan.scan_id }))),
    el("td", {}, when(scan.started), el("br"), quiet(scan.scan_id)),
    el("td", {}, code(scan.target)),
    el("td", {}, code(scan.ports), el("br"),
       quiet(`${scan.port_count.toLocaleString("en-US")} port${scan.port_count === 1 ? "" : "s"}, ` +
             `${scan.protocols.join(" + ").toUpperCase()}, ${ENGINES[scan.engine] || scan.engine}`)),
    el("td", {}, hostsText(scan.hosts)),
    el("td", {}, String(scan.open_ports)),
    el("td", {}, worstText(scan)),
    el("td", {}, scan.source),
    el("td", {}, el("div", { class: "button-group" },
      el("button", { type: "button", class: "btn btn-sm", "data-scan": scan.scan_id,
                     "aria-label": "Show scan " + scan.scan_id }, "Show"),
      el("button", { type: "button", class: "btn btn-sm btn-secondary", "data-delete": scan.scan_id,
                     "aria-label": "Delete scan " + scan.scan_id }, "Delete"))));
}

function listedSays(listed, more) {
  listedText = `${listed} scan${listed === 1 ? "" : "s"} listed${more ? ", older ones below" : ""}.`;
  setContent(byId("scansStatus"), listedText);
}

async function load() {
  try {
    const answer = await getJSON("/api/v1/scans", last ? { before: last } : undefined);
    byId("scansRows").append(...answer.scans.map(scanRow));
    if (answer.scans.length > 0) {
      last = answer.scans[answer.scans.length - 1].scan_id;
    }
    leftOut += answer.left_out;
    const listed = byId("scansRows").children.length;
    show(byId("scansBox"), listed > 0);
    show(byId("scansOlder"), answer.more && answer.scans.length > 0);
    show(byId("compareControls"), listed > 1);
    byId("scansLeftOut").textContent = `${leftOut} file${leftOut === 1 ? " is" : "s are"} in the scan history folder but not a scan this toolkit can show (edited, or from a newer version): left out.`;
    show(byId("scansLeftOut"), leftOut > 0);
    if (listed === 0) {
      message(byId("scansStatus"), "info", "No saved scans yet: each scan that finishes on the Port Scanner page or in the terminal is saved here.");
    } else {
      listedSays(listed, answer.more);
    }
  } catch (error) {
    message(byId("scansStatus"), "error", "Error: " + error.message);
  }
}

// What the summary panel doesn't say: when, from where, and the ports asked for
function about(scan) {
  return `Started ${when(scan.started)}, asked from ${scan.source}. Ports: ${scan.port_list}.`;
}

async function showScan(scanId) {
  message(byId("scansStatus"), "info", "Reading scan " + scanId + "...");
  try {
    const scan = await getJSON("/api/v1/scans/" + encodeURIComponent(scanId));
    setContent(byId("scansStatus"), listedText);
    shown = scanId;
    byId("scanResultsTitle").textContent = "Scan " + scanId;
    byId("scanAbout").textContent = about(scan);
    byId("scanAgain").href = "/scan#again=" + scanId;
    byId("reportOpen").href = "/scans/" + encodeURIComponent(scanId) + "/report";
    for (const [id, fmt] of [["reportHtml", "html"], ["reportTxt", "txt"], ["reportJson", "json"],
                             ["reportCsv", "csv"]]) {
      byId(id).href = "/api/v1/scans/" + encodeURIComponent(scanId) + "/report?format=" + fmt;
    }
    show(byId("comparisonSection"), false);
    showResults(scan, true);
  } catch (error) {
    message(byId("scansStatus"), "error", "Error: " + error.message);
  }
}

// A is the older of the two (a scan id starts with its time)
async function compare() {
  const [a, b] = chosen().sort();
  message(byId("scansStatus"), "info", "Comparing the two scans...");
  try {
    const answer = await getJSON("/api/v1/scans/compare", { a: a, b: b });
    setContent(byId("scansStatus"), listedText);
    hideResults();
    shown = null;
    setContent(byId("comparisonView"), comparisonView(answer));
    show(byId("comparisonSection"));
    byId("comparisonTitle").focus();
  } catch (error) {
    message(byId("scansStatus"), "error", "Error: " + error.message);
  }
}

function askDelete(scanId) {
  deleting = scanId;
  byId("deleteTitle").textContent = "Delete scan " + scanId + "?";
  byId("deleteView").textContent = `Deleting it removes its record, ${scanId}.json, from the scan history on this computer; it can't be undone. A report you downloaded or exported is not touched.`;
  show(byId("deleteSection"));
  byId("deleteTitle").focus();
}

function closeDelete() {
  show(byId("deleteSection"), false);
  deleting = null;
}

// The list without a deleted scan's line, and not the scan shown if it was that one
function dropRow(scanId) {
  byId("scansRows").querySelector(`button[data-delete="${scanId}"]`).closest("tr").remove();
  if (shown === scanId) {
    hideResults();
    shown = null;
  }
  show(byId("comparisonSection"), false);  // it may have compared that scan
  const listed = byId("scansRows").children.length;
  show(byId("scansBox"), listed > 0);
  show(byId("compareControls"), listed > 1);
  choiceChanged();
  listedSays(listed, false);
}

async function confirmDelete() {
  const scanId = deleting;
  closeDelete();
  try {
    await send("/api/v1/scans/" + encodeURIComponent(scanId) + "/delete", "POST", {});
    dropRow(scanId);
    message(byId("scansStatus"), "success", `Scan ${scanId} deleted.`);
  } catch (error) {
    message(byId("scansStatus"), "error", "Error: " + error.message);
  }
  byId("scansStatus").focus();
}

function showFromHash() {
  const scanId = window.location.hash.slice(1);
  if (SCAN_ID.test(scanId)) {
    showScan(scanId);
  }
}

async function setup() {
  byId("scansRows").addEventListener("change", choiceChanged);
  byId("scansRows").addEventListener("click", event => {
    const button = event.target.closest("button[data-scan]");
    if (button) {
      window.location.replace("#" + button.dataset.scan);  // the address names the scan shown
      showScan(button.dataset.scan);
    }
    const remove = event.target.closest("button[data-delete]");
    if (remove) {
      askDelete(remove.dataset.delete);
    }
  });
  byId("deleteConfirm").addEventListener("click", confirmDelete);
  byId("deleteCancel").addEventListener("click", closeDelete);
  byId("scansOlder").addEventListener("click", load);
  byId("compareScans").addEventListener("click", compare);
  await load();
  choiceChanged();
  showFromHash();
}

setup();
