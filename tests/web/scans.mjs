// Runs the saved scans' comparison (scancompare.js) in Node with the minimal DOM of
// fake_dom.mjs, and prints what it shows as JSON for tests/web/test_scans_page.py.
// Usage: node scans.mjs <the js folder>

import { pathToFileURL } from "node:url";
import path from "node:path";

import { installFakeDom } from "./fake_dom.mjs";

installFakeDom();

const folder = process.argv[2];
const { comparisonView } = await import(pathToFileURL(path.join(folder, "scancompare.js")).href);

const XSS = "<img src=x onerror=alert(1)>";
const scan = (scanId, target) => ({
  scan_id: scanId, started: "2026-11-01T10:00:00Z", source: "this computer", target: target,
  ports: "22,80", port_count: 2, protocols: ["tcp"], engine: "builtin",
  hosts: { scanned: 1, skipped: 0, failed: 0 }, open_ports: 1, worst: null,
  worst_unconfirmed: false, exit_code: 0 });
const side = (service, software, severity = "HIGH", maxCvss = 7.5) => ({
  service: service, software: software, severity: severity, severity_unconfirmed: false,
  max_cvss: maxCvss, cves: 1 });
const row = (port, status, a, b, extra = {}) => ({
  port: port, protocol: "tcp", status: status, a: a, b: b, changes: [], cves_added: null,
  cves_gone: null, cves_note: "", ...extra });
const answer = (sameTarget) => ({
  a: scan("20261101T100000Z-0000000a", "10.0.0.5"),
  b: scan("20261102T100000Z-0000000b", sameTarget ? "10.0.0.5" : "10.0.0.4/31"),
  same_target: sameTarget, common: ["10.0.0.5"], only_in_a: [], only_in_b: sameTarget ? [] : ["10.0.0.4"],
  hosts: [
    { address: "10.0.0.5", a: "scanned", b: "scanned", compared: true, rows: [
      row(22, "changed", side("ssh", "OpenSSH 8.9"), side("ssh", XSS), {
        changes: ["software"], cves_added: ["CVE-3"], cves_gone: ["CVE-1"] }),
      row(80, "opened", null, side("http", "nginx 1.25.3")),
      row(443, "closed", side("https", "", "UNKNOWN", null), null),
      row(8080, "not_scanned_in_a", null, side("http", "x")),
      row(25, "same", side("smtp", "Postfix"), side("smtp", "Postfix", "NOT CHECKED", null), {
        cves_note: "not checked in B" }),
    ] },
    { address: "10.0.0.6", a: "scanned", b: "skipped", compared: false, rows: [] },
  ],
  totals: { opened: 1, closed: 1, changed: 1, same: 1, not_scanned: 1, cves_added: 1, cves_gone: 1 },
});

function describe(node) {
  return { tag: node.tagName, class: node.className, text: node.textContent };
}

const holder = document.createElement("div");
const shown = nodes => {
  holder.replaceChildren(...[nodes].flat(Infinity).filter(Boolean));
  return holder.all().slice(1).map(describe);
};

console.log(JSON.stringify({
  same: shown(comparisonView(answer(true))),
  apart: shown(comparisonView(answer(false))),
  nothing: shown(comparisonView({ ...answer(false), common: [], hosts: [] })),
}));
