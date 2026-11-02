// What the saved runs' page (/runs, v5.3 F2) shows, made from the answers of /api/v1/runs: a
// line per run, one run with its leaderboard and charts, and two runs side by side. A
// difference (Δ = B − A) only when both scored the same test rows; otherwise the numbers side
// by side and why. Every value goes in as text (dom.js), so a name from a file stays text.

import { el, scrollBox } from "./dom.js";
import { leaderboardTable, runCharts, seconds } from "./charts.js";
import { balancingName, fmt } from "./format.js";
import { chips, code, quiet, statTile, summaryPanel } from "./summary.js";

export const RUN_ID = /^\d{8}T\d{6}Z-[0-9a-f]{8}$/;  // as the server checks it

const METRICS = [
  ["cv_macro_f1_mean", "CV macro-F1"], ["cv_balanced_accuracy_mean", "CV balanced acc."],
  ["test_macro_f1", "Test macro-F1"], ["test_balanced_accuracy", "Test balanced acc."],
  ["test_accuracy", "Test accuracy"], ["test_min_recall", "Weakest class recall"],
  ["fit_seconds", "Fit time"],
];

export const REASONS = {
  DIFFERENT_DATASET: "They were trained on different datasets (the files' SHA-256 differ).",
  DIFFERENT_TEST_FILE: "Their test rows came from different test files, or only one had a test file.",
  DIFFERENT_LABEL: "They predict different label columns.",
  DIFFERENT_SPLIT: "Their splits differ (the strategy, the seed, the test size or the official mode).",
  DIFFERENT_ROWS: "They had different numbers of training or test rows.",
};

// "2026-10-30T12:00:00Z" as "2026-10-30 12:00 UTC"
export function when(createdAt) {
  const match = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})/.exec(createdAt);
  return match ? `${match[1]} ${match[2]} UTC` : createdAt;
}

function rows(count) {
  return `${Number(count).toLocaleString("en-US")} rows`;
}

// A difference with its sign: "+0.012", "−0.004", "0.000"; seconds for the fit time. One
// that rounds to zero has no sign (not "+0.000")
export function delta(value, key) {
  const size = key === "fit_seconds" ? seconds(Math.abs(value)) : fmt(Math.abs(value), 3);
  if (!/[1-9]/.test(size)) {
    return size;
  }
  return value > 0 ? "+" + size : "−" + size;
}

function metric(value, key) {
  return key === "fit_seconds" ? seconds(value) : fmt(value, 3);
}

// One line of the list: a box to choose it for the comparison, and a button to show it
export function runRow(run) {
  return el("tr", {},
    el("td", {}, el("label", { class: "choice run-pick" },
      el("input", { type: "checkbox", value: run.run_id, "aria-label": "Compare run " + run.run_id }))),
    el("td", {}, when(run.created_at), el("br"), quiet(run.run_id)),
    el("td", {}, code(run.dataset.name || "–"), el("br"), quiet(rows(run.dataset.rows)),
       run.test_dataset && [el("br"), quiet("test file: "), code(run.test_dataset)]),
    el("td", {}, code(run.label)),
    el("td", {}, run.split, el("br"), quiet(`${run.train_rows.toLocaleString("en-US")} / ${run.test_rows.toLocaleString("en-US")}`)),
    el("td", {}, run.selected_title ? "★ " + run.selected_title : "–", el("br"),
       quiet(`of ${run.models.length} model${run.models.length === 1 ? "" : "s"}`)),
    el("td", {}, run.cv_macro_f1_mean === null ? "–" : fmt(run.cv_macro_f1_mean, 3)),
    el("td", {}, run.test_macro_f1 === null ? "–" : fmt(run.test_macro_f1, 3)),
    el("td", {}, el("button", { type: "button", class: "btn btn-sm", "data-run": run.run_id,
                                "aria-label": "Show run " + run.run_id }, "Show")));
}

// What a run was trained on and how, as tiles
function runPanel(id, title, run) {
  return summaryPanel({ id: id, title: title, tiles: [
    statTile("Saved", when(run.created_at)),
    statTile("Dataset", code(run.dataset.name || "–"), rows(run.dataset.rows)),
    run.test_dataset ? statTile("Test file", code(run.test_dataset)) : null,
    run.profile ? statTile("Profile", run.profile) : null,
    statTile("Label", code(run.label)),
    statTile("Split", run.split, `${run.train_rows.toLocaleString("en-US")} training, ${run.test_rows.toLocaleString("en-US")} test rows`),
    statTile("Balancing", balancingName(run.balancing)),
    statTile("Selected model", run.selected_title || "None", run.selected_title &&
             `CV macro-F1 ${fmt(run.cv_macro_f1_mean, 3)}, test ${fmt(run.test_macro_f1, 3)}`),
  ] });
}

// One saved run: what it was, its leaderboard and its charts
export function runView(view) {
  return [
    runPanel("savedRun", "Run " + view.summary.run_id, view.summary),
    el("h3", {}, "Leaderboard"),
    el("p", {}, "Ranked by CV macro-F1; ★ = the selected model."),
    leaderboardTable(view.leaderboard),
    runCharts(view.leaderboard, view.folds),
  ];
}

// A setting's value as the comparison shows it
function settingValue(item, side, titles) {
  const value = item[side];
  if (item.setting === "features") {
    const other = new Set(item[side === "a" ? "b" : "a"]);
    const only = value.filter(name => !other.has(name));
    return [`${value.length} features`, only.length > 0 && [el("br"), quiet("only here: "), chips(only)]];
  }
  if (item.setting === "models") {
    return value.map(name => titles[name] || name).join(", ");
  }
  if (item.setting === "balancing") {
    return balancingName(value);
  }
  const [selector, k] = value;  // selection
  return selector === "none" ? "None" : k === null ? selector : `${selector}, ${k} features`;
}

const SETTING_NAMES = { features: "Features", models: "Models", balancing: "Balancing",
                        selection: "Feature selection" };

// One model of both runs: every number of A and B, their difference when comparable, and
// the test recall of the classes in both
function modelTable(model, comparable) {
  const star = model.selected[0] && model.selected[1] ? " (★ selected in both)" :
    model.selected[0] ? " (★ selected in A)" : model.selected[1] ? " (★ selected in B)" : "";
  const headers = ["Number", "A", "B", comparable && "Δ (B − A)"].filter(text => text !== false);
  const line = (name, a, b, d) => el("tr", {}, el("th", { scope: "row" }, name),
    el("td", {}, a), el("td", {}, b), comparable && el("td", {}, d));
  return el("figure", { class: "chart compare-model" },
    el("figcaption", {}, model.title + star),
    scrollBox(model.title + ": A and B", el("table", { class: "metrics-table compare-table" },
      el("thead", {}, el("tr", {}, headers.map(text => el("th", { scope: "col" }, text)))),
      el("tbody", {},
        METRICS.map(([key, name]) => line(name, metric(model.a[key], key), metric(model.b[key], key),
                                          comparable && delta(model.delta[key], key))),
        model.recall.length > 0 && el("tr", {}, el("th", { scope: "rowgroup", colspan: headers.length,
                                                           class: "compare-group" }, "Test recall per class")),
        model.recall.map(item => line(code(item.label), fmt(item.a, 3), fmt(item.b, 3),
                                      comparable && delta(item.delta, "recall")))))));
}

// Two saved runs side by side (A the older)
export function comparisonView(answer) {
  const titles = answer.titles;
  const oneRun = (names, run) => names.length > 0 &&
    el("p", {}, `Only in ${run}: `, names.map(name => titles[name] || name).join(", "), ".");
  return [
    el("div", { class: "compare-runs" },
      runPanel("runA", "A: run " + answer.a.run_id, answer.a),
      runPanel("runB", "B: run " + answer.b.run_id, answer.b)),
    answer.comparable ?
      el("p", { class: "info-message" }, "Both runs scored the same test rows, so their numbers can be compared: Δ = B − A.") :
      el("div", { class: "warning-message" },
        el("p", {}, "These runs did not score the same test rows, so the numbers are side by side, without a difference:"),
        el("ul", {}, answer.reasons.map(reason => el("li", {}, REASONS[reason] || reason)))),
    el("h3", {}, "Each model of both runs"),
    answer.models.length === 0 ? el("p", {}, "The two runs have no model in common.") :
      el("div", { class: "compare-models" }, answer.models.map(model => modelTable(model, answer.comparable))),
    oneRun(answer.only_in_a, "A"), oneRun(answer.only_in_b, "B"),
    el("h3", {}, "Settings that differ"),
    answer.settings.length === 0 ? el("p", {}, "None: both runs used the same features, models, balancing and feature selection.") :
      scrollBox("Settings that differ", el("table", { class: "metrics-table" },
        el("thead", {}, el("tr", {}, ["Setting", "A", "B"].map(text => el("th", { scope: "col" }, text)))),
        el("tbody", {}, answer.settings.map(item => el("tr", {},
          el("th", { scope: "row" }, SETTING_NAMES[item.setting] || item.setting),
          el("td", {}, settingValue(item, "a", titles)), el("td", {}, settingValue(item, "b", titles))))))),
  ];
}
