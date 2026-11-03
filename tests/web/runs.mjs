// Runs the saved runs' page code (runsview.js) in Node with the minimal DOM of fake_dom.mjs,
// and prints what it shows as JSON for tests/web/test_runs_page.py.
// Usage: node runs.mjs <the js folder>

import { pathToFileURL } from "node:url";
import path from "node:path";

import { installFakeDom } from "./fake_dom.mjs";

installFakeDom();

const folder = process.argv[2];
const { comparisonView, deleteView, delta, fileSize, runRow, runView, storageView, when } =
  await import(pathToFileURL(path.join(folder, "runsview.js")).href);

const XSS = "<img src=x onerror=alert(1)>";
const run = (runId, extra = {}) => ({
  run_id: runId, created_at: "2026-10-30T12:05:09Z", dataset: { name: XSS, rows: 12345 },
  test_dataset: null, profile: null, label: "Label", split: "stratified", train_rows: 8641,
  test_rows: 3704, balancing: "none", models: ["random_forest", "decision_tree"],
  selected: "random_forest", selected_title: "Random Forest", cv_macro_f1_mean: 0.9123,
  test_macro_f1: 0.9444, bundle: "exported", ...extra,
});
const A = "20261030T120509Z-0000000a";
const B = "20261031T090000Z-0000000b";
const board = [
  { model: "random_forest", title: "Random Forest", selected: true, cv_macro_f1_mean: 0.912,
    cv_macro_f1_std: 0.004, test_macro_f1: 0.944, test_balanced_accuracy: 0.95, test_accuracy: 0.99,
    test_min_recall: 0.8, fit_seconds: 12.5, test_recall_per_class: [["BENIGN", 0.99], [XSS, 0.8]] },
];
const numbers = (shift) => ({ cv_macro_f1_mean: 0.9 + shift, cv_balanced_accuracy_mean: 0.91 + shift,
  test_macro_f1: 0.92 + shift, test_balanced_accuracy: 0.93 + shift, test_accuracy: 0.94 + shift,
  test_min_recall: 0.5 + shift, fit_seconds: 10 + shift * 100 });
const model = (comparable) => ({
  model: "random_forest", title: "Random Forest", selected: [true, false],
  a: numbers(0), b: numbers(0.012),
  delta: comparable ? { cv_macro_f1_mean: 0.012, cv_balanced_accuracy_mean: 0.012, test_macro_f1: 0.012,
                        test_balanced_accuracy: 0.012, test_accuracy: 0.012, test_min_recall: -0.004,
                        fit_seconds: 1.2 } : null,
  recall: [{ label: XSS, a: 0.8, b: 0.75, delta: comparable ? -0.05 : null }],
});
const answer = (comparable) => ({
  a: run(A), b: run(B, { test_dataset: "KDDTest+.csv", profile: "NSL-KDD" }),
  comparable: comparable, reasons: comparable ? [] : ["DIFFERENT_DATASET", "DIFFERENT_ROWS", "NEW_CODE"],
  models: [model(comparable)], only_in_a: ["decision_tree"], only_in_b: ["xgboost"],
  titles: { random_forest: "Random Forest", decision_tree: "Decision Tree" },
  settings: [
    { setting: "features", a: ["f1", "f2", XSS], b: ["f1", "f2"] },
    { setting: "models", a: ["random_forest", "decision_tree"], b: ["random_forest", "xgboost"] },
    { setting: "balancing", a: "none", b: "smote" },
    { setting: "selection", a: ["none", null], b: ["mutual_info", 20] },
  ],
});

// What a test needs of an element: its tag, class, attributes and text
function describe(node) {
  return { tag: node.tagName, class: node.className, attributes: Object.fromEntries(node.attributes),
           text: node.textContent };
}

const holder = document.createElement("div");
const shown = nodes => {
  holder.replaceChildren(...[nodes].flat(Infinity).filter(Boolean));
  return holder.all().slice(1).map(describe);
};

console.log(JSON.stringify({
  row: shown(runRow(run(A))),
  row_cells: runRow(run(A, { test_dataset: "KDDTest+.csv" })).children.map(cell => cell.textContent),
  bundle_cells: [null, "exported", "incomplete", "unreadable"].map(bundle =>
    runRow(run(A, { bundle })).children[8].textContent),
  view: shown(runView({ summary: run(A), leaderboard: board, folds: { random_forest: [0.91, 0.912, 0.914] } },
                     { importances: [], left_out: 0 })),
  comparable: shown(comparisonView(answer(true))),
  apart: shown(comparisonView(answer(false))),
  no_common: shown(comparisonView({ ...answer(true), models: [], settings: [] })),
  deltas: [[0.012], [-0.004], [0], [0.0002], [-0.0004], [1.2, "fit_seconds"], [-4414, "fit_seconds"]].map(([value, key]) => delta(value, key)),
  when: [when("2026-10-30T12:05:09Z"), when("not a time")],
  deleting: shown(deleteView({ run_id: A, files: [
    { kind: "importance", name: `${A}.importance.knn.json`, size: 2100 },
    { kind: "bundle", name: `${A}.bundle`, size: 1980268869 },
    { kind: "record", name: XSS, size: 812 }] })),
  deleting_no_bundle: shown(deleteView({ run_id: A, files: [
    { kind: "record", name: `${A}.json`, size: 812 }] })),
  sizes: [0, 812, 1024, 2100, 40749566, 1980268869].map(fileSize),
  storage: shown(storageView({ total_bytes: 1980270000, free_bytes: 5 * 2 ** 30, warnings: [XSS], kinds: [
    { kind: "records", title: "Saved runs' records", count: 1, bytes: 812, files: [], leftover: false },
    { kind: "scans", title: "Scan reports", count: 0, bytes: 0, files: [], leftover: false },
    { kind: "unregistered_bundles", title: "Bundles whose run was deleted", count: 1, bytes: 1980268869,
      files: [`${A}.bundle`], leftover: true },
    { kind: "damaged", title: "Damaged files", count: 1, bytes: 319, files: [XSS], leftover: true }] })),
  storage_empty: shown(storageView({ total_bytes: 0, free_bytes: 2 ** 30, warnings: [], kinds: [] })),
}));
