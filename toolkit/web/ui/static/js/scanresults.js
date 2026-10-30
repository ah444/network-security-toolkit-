// A finished scan's answer (v5.2 step 2.4e): a summary panel, the hosts and a card per open
// port (its details open on demand). Banners, software names, CPE names, AbuseIPDB's fields
// and the AI's text come from the network: they go in as text only (dom.js). CVE IDs are
// plain text, never links: the page opens no other site.

import { byId, el, setContent, show } from "./dom.js";
import { code, quiet, statTile, summaryPanel } from "./summary.js";

const CVES_SHOWN = 10;
const SEVERITY_CLASS = { CRITICAL: "critical", HIGH: "high", MEDIUM: "medium", LOW: "low",
                         "NO KNOWN CVE": "none", UNKNOWN: "unknown" };
const HOST_STATES = { scanned: "scanned", skipped: "skipped: it didn't answer ping",
                      failed: "the engine couldn't scan it" };

// "cvss-" + the level: the card's edge and the level's colour (app.css); the word is always
// written too, so the level never rests on the colour alone
function severityClass(severity) {
  return "cvss-" + (SEVERITY_CLASS[severity] || "unknown");
}

function plural(count, word) {
  return count + " " + word + (count === 1 ? "" : "s");
}

// The requests sent to each online service: NVD, the AI, AbuseIPDB, then any other host
function sentText(sent) {
  const order = ["NVD", "AI (OpenRouter)", "AbuseIPDB"];
  const names = order.filter(name => name in sent).concat(
    Object.keys(sent).filter(name => !order.includes(name)));
  return names.map(name => name + " " + sent[name]).join(", ");
}

function summary(answer) {
  const totals = answer.summary;
  const online = answer.online;
  return summaryPanel({
    id: "scanSummaryPanel",
    title: answer.message,
    tiles: [
      statTile("Target", code(answer.target)),
      statTile("Engine", answer.engine === "nmap" ? "nmap" : "Built-in",
               answer.engine === "nmap" ? answer.engine_notice[0] : null),
      statTile("Ports", plural(answer.ports, "port"), answer.protocols.join(" + ").toUpperCase()),
      statTile("Time", totals.seconds + " s"),
      statTile("Sent online", sentText(totals.sent) || "nothing"),
      statTile("Failed lookups", String(totals.failed_lookups.length),
               totals.failed_lookups.join("; ") || null),
    ],
    foot: online && "What could go online: NVD " + online.nvd + "; AI " + online.ai +
      "; AbuseIPDB " + online.abuseipdb + ".",
  });
}

function hostItem(host, reputation) {
  const found = reputation[host.target];
  return el("li", {},
    code(host.target), ": ", HOST_STATES[host.state] || host.state,
    host.os && el("span", { class: "host-os" }, "OS: " + host.os),
    found && el("span", { class: "host-os" }, found.error ? "AbuseIPDB: the lookup failed (" +
      found.error + ")" : "AbuseIPDB: confidence of abuse " + found.abuseConfidenceScore +
      "%, " + found.totalReports + " report(s), ISP " + found.isp + ", country " +
      found.countryCode));
}

function detail(label, value) {
  return value ? [el("dt", {}, label), el("dd", {}, value)] : [];
}

// The worst CVEs first, one a line (at most CVES_SHOWN; the download has all of them)
function cveList(entry) {
  const cves = entry.cves;
  if (cves.length === 0) {
    return entry.software ? "none found" : null;
  }
  return el("ul", { class: "cve-list" },
    cves.slice(0, CVES_SHOWN).map(cve => el("li", {}, el("span", { class: "mono" }, cve.id),
      " · CVSS " + (cve.cvss === null ? "?" : cve.cvss) + " · " + cve.severity)),
    cves.length > CVES_SHOWN && el("li", { class: "quiet" }, "and " + (cves.length - CVES_SHOWN) +
      " more: all of them are in the downloaded report"));
}

function portCard(entry, manyHosts) {
  const level = severityClass(entry.severity);
  const software = entry.software || "software not recognised";
  return el("details", { class: "port-card " + level },
    el("summary", {},
      el("span", { class: "mono" }, (manyHosts ? entry.target + " " : "") + entry.port + "/" +
         entry.protocol),
      " ", entry.service, " · ", software, " · ",
      el("span", { class: "cvss-level " + level }, entry.software ? entry.severity : "no lookup"),
      entry.software && " · " + plural(entry.cves.length, "CVE")),
    el("dl", { class: "port-details" },
      detail("Banner", entry.banner && code(entry.banner)),
      detail("CPE (nmap)", entry.cpe.length > 0 && code(entry.cpe.join(" "))),
      detail("Software", entry.software ? entry.software :
        quiet(entry.engine === "nmap" ? "not recognised by nmap or in the banner (no CVE lookup)" :
              "not recognised in the banner (no CVE lookup)")),
      detail("CVEs (NVD)", cveList(entry)),
      detail("Severity (from CVSS)", entry.software && entry.severity +
        (entry.max_cvss === null ? "" : " (highest CVSS " + entry.max_cvss + ")")),
      detail("CVE source", entry.cve_source),
      detail("AI explanation (may be wrong)", entry.ai_explanation &&
        el("span", { class: "ai-text" }, entry.ai_explanation)),
      detail("Lookups that failed", entry.lookup_errors.join("; "))));
}

// Show a finished scan's answer; ``focus`` moves the keyboard to its heading
export function showResults(answer, focus) {
  setContent(byId("scanSummary"), summary(answer));
  setContent(byId("scanHosts"), answer.hosts.map(host => hostItem(host, answer.reputation)));
  const manyHosts = new Set(answer.entries.map(entry => entry.target)).size > 1;
  setContent(byId("scanPortCards"), answer.entries.length === 0 ?
    el("p", {}, "No open ports found.") :
    answer.entries.map(entry => portCard(entry, manyHosts)));
  show(byId("scanResults"));
  if (focus) {
    byId("scanResultsTitle").focus();
  }
}

export function hideResults() {
  show(byId("scanResults"), false);
}
