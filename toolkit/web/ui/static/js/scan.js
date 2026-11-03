// Port scanner page (/scan, v5.2 step 2.4e), loaded as an ES module. The form's choices come
// from GET /api/v1/scan/options; the scan runs as a job on the server (scanrun.js follows it)
// and its answer is shown by scanresults.js. The authorization names the target and is
// asked again for every new target; the server refuses a scan without it, and sends nothing
// at all before it (docs/web_api.md, "Port scan"). A finished scan says where it was saved
// (F6a); "#again=<scan id>" (Saved Scans' Scan again, F6b) fills the form with that scan's
// choices, the authorization unticked.

import { getJSON, send } from "./api.js";
import { byId, el, fillSelect, message, setContent, show } from "./dom.js";
import { follow, setup as setupRun, starting } from "./scanrun.js";
import { hideResults, showResults } from "./scanresults.js";

const SCAN_ID = /^\d{8}T\d{6}Z-[0-9a-f]{8}$/;

let options = null;

function plural(count, word) {
  return count.toLocaleString() + " " + word + (count === 1 ? "" : "s");
}

// The form's choices: nmap (when it can run here) and its settings, and which online
// lookups have a key on the server
function fillOptions() {
  const limits = options.limits;
  byId("scanLimits").textContent = "Up to " + plural(limits.max_hosts, "host") + " and " +
    plural(limits.max_checks, "host × port check") + "; a scan stops after " +
    plural(limits.max_minutes, "minute") + ".";
  const nmap = options.engines.find(engine => engine.id === "nmap");
  byId("engineNmap").disabled = !nmap.available;
  byId("nmapStatus").textContent = !nmap.available ? nmap.reason : nmap.raw_packets ?
    "nmap is installed and can send raw packets (a SYN scan, UDP and OS detection)." :
    "nmap is installed; without raw packets it makes a TCP connect scan (no UDP, no OS detection).";
  const choices = options.nmap;
  fillSelect(byId("nmapDetection"), choices.service_detection.map(level =>
    ({ value: level, text: level, selected: level === choices.default_service_detection })));
  fillSelect(byId("nmapTiming"), choices.timing.map(timing =>
    ({ value: timing, text: timing, selected: timing === choices.default_timing })));
  byId("nmapScriptNames").textContent = "The toolkit's " + choices.scripts.length + ": " +
    choices.scripts.join(", ") + ". Each talks to the target only.";
  byId("nmapOsDetect").disabled = !choices.udp_and_os_detection;
  byId("nmapRawNote").textContent = choices.raw_packets_note;
  show(byId("nmapRawNote"), !choices.udp_and_os_detection);
  byId("nmapVersionNote").textContent = choices.version_scripts_note;
  // On by default (HS4b, RD5); without its key a service is off and its switch can't be
  // turned on until the key is in .env and the page is loaded again
  for (const [box, note, key, variable] of [["scanAbuse", "scanAbuseNote", "abuseipdb", "ABUSEIPDB_API_KEY"],
                                            ["scanAi", "scanAiNote", "openrouter", "OPENROUTER_API_KEY"]]) {
    byId(box).disabled = !options.keys[key];
    byId(box).checked = options.keys[key];
    byId(note).textContent = options.keys[key] ? "" :
      "No API key: add " + variable + " to .env, then refresh this page.";
    show(byId(note), !options.keys[key]);
  }
  // NVD needs no key: without one it is asked more slowly
  byId("scanNvdNote").textContent = options.keys.nvd ? "" :
    "No NVD_API_KEY: NVD is asked more slowly. To go faster, add it to .env, then refresh this page.";
  show(byId("scanNvdNote"), !options.keys.nvd);
  engineChanged();
}

// nmap's choices only with nmap; with nmap and no raw packets, TCP only
function engineChanged() {
  const nmap = byId("engineNmap").checked;
  show(byId("nmapOptions"), nmap);
  const tcpOnly = nmap && options !== null && !options.nmap.udp_and_os_detection;
  byId("protocolUdp").disabled = tcpOnly;
  byId("protocolBoth").disabled = tcpOnly;
  if (tcpOnly) {
    byId("protocolTcp").checked = true;
  }
}

// An IPv4 address or network outside the private, loopback and link-local ranges (a host
// name can't be told without a DNS query, which the page never makes)
function looksPublic(target) {
  const match = /^(\d{1,3})\.(\d{1,3})\.\d{1,3}\.\d{1,3}(\/\d{1,2})?$/.exec(target);
  if (!match) {
    return false;
  }
  const [a, b] = [Number(match[1]), Number(match[2])];
  const local = a === 10 || a === 127 || a === 0 || (a === 172 && b >= 16 && b <= 31) ||
    (a === 192 && b === 168) || (a === 169 && b === 254) || (a === 100 && b >= 64 && b <= 127);
  return !local && a < 224;
}

// A new target asks for the authorization again, naming it
function targetChanged() {
  const target = byId("scanTarget").value.trim();
  byId("scanAuthorized").checked = false;
  byId("scanAuthorizedText").textContent = target ?
    "I own " + target + " or have written permission to scan it." :
    "I own this target or have written permission to scan it.";
  show(byId("publicTargetWarning"), looksPublic(target));
}

function choice(name) {
  return document.querySelector('input[name="' + name + '"]:checked').value;
}

function request() {
  const body = {
    target: byId("scanTarget").value.trim(),
    ports: byId("scanPortList").value.trim(),
    protocol: choice("protocol"),
    engine: choice("engine"),
    scan_unresponsive: byId("scanUnresponsive").checked,
    nvd: byId("scanNvd").checked,
    abuseipdb: byId("scanAbuse").checked,
    ai: byId("scanAi").checked,
    authorized: byId("scanAuthorized").checked,
  };
  if (body.engine === "nmap") {
    Object.assign(body, {
      service_detection: byId("nmapDetection").value,
      timing: byId("nmapTiming").value,
      scripts: byId("nmapScripts").checked,
      os_detect: byId("nmapOsDetect").checked,
    });
  }
  return body;
}

// Where a finished scan was saved (F6a), with a 44 px button-like link to it, or why it wasn't
function showSaved(answer) {
  const saved = answer.saved || {};
  const line = byId("scanSaved");
  if (saved.scan_id) {
    setContent(line, el("p", {}, "Saved in the scan history as " + saved.scan_id + "."),
               el("a", { href: "/scans#" + answer.saved.scan_id, class: "btn btn-secondary btn-sm" },
                  "See it in Saved Scans"));
  } else {
    setContent(line, saved.error ? el("p", {}, "Not saved in the scan history: " + saved.error) : "");
  }
  line.className = saved.error ? "warning-message" : "field-hint";
  show(line, Boolean(saved.scan_id || saved.error));
}

function showScan(answer, focus) {
  showResults(answer, focus);
  showSaved(answer);
}

// The last finished scan of the session, if any: the one the download gives
async function showLastScan() {
  const scan = await getJSON("/api/v1/scan");
  if (scan.last) {
    showScan(scan.last, false);
  }
}

// Saved Scans' Scan again: that scan's choices in the form; the authorization is unticked and
// names the target, as for any new target (a choice this computer can't make stays off)
async function scanAgain(scanId) {
  const status = byId("scanStatus");
  try {
    const scan = await getJSON("/api/v1/scans/" + encodeURIComponent(scanId));
    const again = scan.again;
    byId("scanTarget").value = again.target;
    byId("scanPortList").value = again.ports;
    const engine = again.engine === "nmap" && !byId("engineNmap").disabled ? "nmap" : "builtin";
    byId(engine === "nmap" ? "engineNmap" : "engineBuiltin").checked = true;
    engineChanged();
    if (engine === "nmap") {
      byId("nmapDetection").value = again.service_detection;
      byId("nmapTiming").value = again.timing;
      byId("nmapScripts").checked = again.scripts !== "";
      byId("nmapOsDetect").checked = again.os_detect && !byId("nmapOsDetect").disabled;
    }
    const protocol = { tcp: "protocolTcp", udp: "protocolUdp", both: "protocolBoth" }[again.protocol];
    if (!byId(protocol).disabled) {
      byId(protocol).checked = true;
    }
    byId("scanUnresponsive").checked = again.scan_unresponsive;
    byId("scanNvd").checked = again.nvd;
    for (const [box, on] of [["scanAbuse", again.abuseipdb], ["scanAi", again.ai]]) {
      byId(box).checked = on && !byId(box).disabled;
    }
    targetChanged();
    message(status, "info", "The choices of scan " + scanId + " are filled in" +
      (engine !== again.engine ? ", with the built-in engine: nmap can't run here" : "") +
      ". Check them, tick the authorization, then start the scan.");
    byId("scanTarget").focus();
  } catch (error) {
    message(status, "error", "Error: " + error.message);
  }
}

// A scan ended: its answer, or (cancelled or failed: nothing kept) the last finished one again
function scanEnded(answer, focus) {
  if (answer !== null) {
    showScan(answer, focus);
  } else {
    showLastScan().catch(error => console.error("Could not show the last scan:", error));
  }
}

async function start(event) {
  event.preventDefault();
  const status = byId("scanStatus");
  starting();
  try {
    const job = await send("/api/v1/jobs/scan", "POST", request());
    hideResults();
    follow(job, scanEnded, true);
  } catch (error) {  // refused: nothing was queued
    byId("scanStart").disabled = false;
    message(status, "error", "Error: " + error.message);
    status.focus();
  }
}

// After a reload: the scan still running, or else the last finished one
async function resume() {
  try {
    const scan = await getJSON("/api/v1/scan");
    if (scan.job) {
      follow(scan.job, scanEnded, false);
    } else if (scan.last) {
      showScan(scan.last, false);
    }
  } catch (error) {
    console.error("Could not resume the scan:", error);
  }
}

async function setup() {
  byId("scanTarget").addEventListener("input", targetChanged);
  document.querySelectorAll('input[name="engine"]').forEach(radio =>
    radio.addEventListener("change", engineChanged));
  byId("scanForm").addEventListener("submit", start);
  setupRun();
  try {
    options = await getJSON("/api/v1/scan/options");
    fillOptions();
  } catch (error) {
    message(byId("scanStatus"), "error", "Error: " + error.message);
  }
  resume();
  const again = window.location.hash.replace(/^#again=/, "");
  if (options !== null && SCAN_ID.test(again)) {
    scanAgain(again);
  }
}

setup();
