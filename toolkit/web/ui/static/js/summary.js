// Result panels: a heading with an icon, label/value tiles in a <dl> and a footnote
// (the Upload Dataset card; the scan results of step 2.4e use the same pattern), and
// chips for a list of names. Every value goes in as text (dom.js), so names from a file
// (a file, a dataset, a column called <img onerror=...>) show as those characters.

import { el, icon } from "./dom.js";

export const MAX_CHIPS = 6;

// A name from the data, in a fixed-width font; dir="auto" lays out Arabic names right to left
export function code(text) {
  return el("span", { class: "mono", dir: "auto" }, text);
}

// A remark next to a value, e.g. "(full)" or "None"
export function quiet(text) {
  return el("span", { class: "quiet" }, text);
}

// One tile: the label above the value (text or nodes), and an optional muted second line
export function statTile(label, value, note = null) {
  return el("div", { class: "stat-tile" },
    el("dt", {}, label),
    el("dd", {}, value, note !== null && el("span", { class: "stat-note" }, note)));
}

// A panel: "success" (a check icon) or "error" (an alert icon, announced at once).
// Tiles that are null are left out.
export function summaryPanel({ id, kind = "success", title, tiles = [], body = [], foot = null }) {
  const error = kind === "error";
  const shown = tiles.filter(tile => tile !== null);
  return el("section", {
    class: error ? "summary-panel summary-panel-error" : "summary-panel",
    "aria-labelledby": id + "Title",
    role: error ? "alert" : null,
  },
    el("h4", { class: "summary-head", id: id + "Title" }, icon(error ? "alert" : "check"), title),
    shown.length > 0 && el("dl", { class: "stat-grid" }, shown),
    body,
    foot !== null && el("p", { class: "summary-foot" }, foot));
}

// A chip per name, at most MAX_CHIPS: with more, the first MAX_CHIPS - 1 and "+N more".
// Each keeps its full name as a tooltip, as a long one is cut with "...".
export function chips(names) {
  const shown = names.length > MAX_CHIPS ? names.slice(0, MAX_CHIPS - 1) : names;
  return [
    shown.map(name => el("span", { class: "chip", title: name, dir: "auto" }, name)),
    names.length > shown.length && el("span", { class: "chip chip-more" },
                                      `+${names.length - shown.length} more`),
  ];
}
