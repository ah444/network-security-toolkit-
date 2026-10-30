// Port scanner page (/scan, v5.2 step 2.4e), loaded as an ES module. The form's choices come
// from GET /api/v1/scan/options; the scan runs as a job on the server (scanrun.js follows it)
// and its answer is shown by scanresults.js. The authorization names the target and is
// asked again for every new target; the server refuses a scan without it, and sends nothing
// at all before it (docs/web_api.md, "Port scan").

import { getJSON, send } from "./api.js";
import { byId, fillSelect, message, show } from "./dom.js";
import { follow, setup as setupRun, starting } from "./scanrun.js";
import { hideResults, showResults } from "./scanresults.js";

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
  for (const [box, note, key, variable] of [["scanAbuse", "scanAbuseNote", "abuseipdb", "ABUSEIPDB_API_KEY"],
                                            ["scanAi", "scanAiNote", "openrouter", "OPENROUTER_API_KEY"]]) {
    byId(box).disabled = !options.keys[key];
    byId(note).textContent = options.keys[key] ? "" : "Off: no " + variable + " is set on the server.";
    show(byId(note), !options.keys[key]);
  }
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

// The last finished scan of the session, if any: the one the download gives
async function showLastScan() {
  const scan = await getJSON("/api/v1/scan");
  if (scan.last) {
    showResults(scan.last, false);
  }
}

// A scan ended: its answer, or (cancelled or failed: nothing kept) the last finished one again
function scanEnded(answer, focus) {
  if (answer !== null) {
    showResults(answer, focus);
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
      showResults(scan.last, false);
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
}

setup();
