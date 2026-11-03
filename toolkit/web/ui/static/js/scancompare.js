// Two saved scans compared (v5.4 F6c), as the Saved Scans page shows them: A is the older. A
// port is opened or closed only when the other scan checked it; CVEs are compared only when
// both scans looked them up. Banners, services and software come from the network: they go in
// as text only (dom.js).

import { el, scrollBox } from "./dom.js";
import { code, quiet } from "./summary.js";

const HOST_STATES = { scanned: "scanned", skipped: "skipped: it didn't answer ping",
                      failed: "the engine couldn't scan it" };
const STATUS = { opened: "opened", closed: "closed", changed: "changed", same: "the same",
                 not_scanned_in_a: "open in B; A didn't check this port",
                 not_scanned_in_b: "open in A; B didn't check this port" };
const CHANGE_NAMES = { service: "service", software: "software", severity: "severity",
                       max_cvss: "highest CVSS" };

function plural(count, word) {
  return count + " " + word + (count === 1 ? "" : "s");
}

function when(started) {
  const match = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})/.exec(started);
  return match ? `${match[1]} ${match[2]} UTC` : started;
}

function scanLine(name, scan) {
  return el("li", {}, `${name}: scan ${scan.scan_id}, ${when(scan.started)}, `, code(scan.target),
    `, ports ${scan.ports} (${scan.protocols.join(" + ").toUpperCase()}), asked from ${scan.source}`);
}

// One scan's side of a port: its service, software and severity; "–" when it wasn't open
function side(found) {
  if (found === null) {
    return "–";
  }
  const severity = found.software ? found.severity + (found.severity_unconfirmed ? " (unconfirmed)" : "") +
    (found.max_cvss === null ? "" : `, highest CVSS ${found.max_cvss}`) : "no lookup";
  return [found.service || quiet("no service"), " · ", found.software || quiet("software not recognised"),
          " · ", severity];
}

function cveCell(row) {
  if (row.cves_added === null) {
    return row.cves_note ? quiet(row.cves_note) : "–";
  }
  if (row.cves_added.length === 0 && row.cves_gone.length === 0) {
    return "no change";
  }
  return [row.cves_added.length > 0 && el("span", {}, "added: ", code(row.cves_added.join(", "))),
          row.cves_added.length > 0 && row.cves_gone.length > 0 && el("br"),
          row.cves_gone.length > 0 && el("span", {}, "gone: ", code(row.cves_gone.join(", ")))];
}

function statusText(row) {
  const words = STATUS[row.status] || row.status;
  return row.changes.length > 0 ? `${words}: ${row.changes.map(name => CHANGE_NAMES[name] || name).join(", ")}` : words;
}

function hostView(host) {
  const heading = el("h4", {}, code(host.address));
  if (!host.compared) {
    return [heading, el("p", {}, `Not compared: in A it was ${HOST_STATES[host.a] || host.a}, in B ${HOST_STATES[host.b] || host.b}.`)];
  }
  if (host.rows.length === 0) {
    return [heading, el("p", {}, "No open port in either scan.")];
  }
  return [heading, scrollBox("Ports of " + host.address, el("table", { class: "metrics-table" },
    el("thead", {}, el("tr", {}, ["Port", "What changed", "A", "B", "CVEs"].map(text => el("th", { scope: "col" }, text)))),
    el("tbody", {}, host.rows.map(row => el("tr", {},
      el("th", { scope: "row", class: "mono" }, `${row.port}/${row.protocol}`),
      el("td", {}, statusText(row)), el("td", {}, side(row.a)), el("td", {}, side(row.b)),
      el("td", {}, cveCell(row)))))))];
}

export function comparisonView(answer) {
  const totals = answer.totals;
  const only = (addresses, name) => addresses.length > 0 &&
    el("p", {}, `Only in ${name}: `, code(addresses.join(", ")), ` (not compared).`);
  return [
    el("ul", {}, scanLine("A, the older", answer.a), scanLine("B, the newer", answer.b)),
    !answer.same_target && el("p", { class: "warning-message" },
      "The two scans had different targets: only the addresses both scanned are compared."),
    only(answer.only_in_a, "A"), only(answer.only_in_b, "B"),
    answer.common.length === 0 ? el("p", {}, "No address in common: there is nothing to compare.") : [
      el("p", { class: "info-message" },
        `${plural(totals.opened, "port")} opened, ${totals.closed} closed, ${totals.changed} changed, ` +
        `${totals.same} the same` + (totals.not_scanned > 0 ? `, ${totals.not_scanned} open in one scan that the other didn't check` : "") +
        `; CVEs: ${totals.cves_added} added, ${totals.cves_gone} gone. A port is closed only when the newer scan checked it, ` +
        "and CVEs are compared only when both scans looked them up."),
      answer.hosts.map(hostView)],
  ];
}
