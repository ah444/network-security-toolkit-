// Runs the page's importance view (charts.js), a saved run's view (runsview.js) and a
// prediction's "Why?" (explain.js) in Node
// with the minimal DOM of fake_dom.mjs, and prints what they show as JSON for
// tests/web/test_importance_page.py.
// Usage: node importance.mjs <the js folder>

import { pathToFileURL } from "node:url";
import path from "node:path";

import { installFakeDom } from "./fake_dom.mjs";

installFakeDom();

const folder = process.argv[2];
const { importanceView } = await import(pathToFileURL(path.join(folder, "charts.js")).href);
const { runView } = await import(pathToFileURL(path.join(folder, "runsview.js")).href);
const { whyView } = await import(pathToFileURL(path.join(folder, "explain.js")).href);

const XSS = "<img src=x onerror=alert(1)>";
const LONG = "Total Length of Bwd Packets Per Second Averaged";
// 20 features, ranked as the server ranks them: the biggest mean drop first; from the
// 14th on, below 0 (shuffling helped)
const features = Array.from({ length: 20 }, (_, i) => ({
  feature: i === 0 ? XSS : i === 1 ? LONG : `feature ${i}`,
  mean: 0.2 - i * 0.016, std: 0.004,
}));
const result = (extra = {}) => ({
  run_id: "20261103T120920Z-5a8b6862", model: "random_forest", rows: 5000, test_rows: 22544,
  repeats: 5, seed: 42, baseline_macro_f1: 0.91, features, concentrated: null, ...extra,
});

function describe(node) {
  const attributes = Object.fromEntries(node.attributes || []);
  return { tag: node.tagName, class: node.className || attributes.class || "",
           attributes, text: node.textContent };
}

function shown(view) {
  const elements = view.all();
  const drawing = elements.find(item => item.tagName === "SVG");
  return {
    tag: view.tagName,
    caption: (elements.find(item => item.tagName === "FIGCAPTION") || { textContent: null }).textContent,
    drawing: drawing ? describe(drawing) : null,
    marks: drawing ? drawing.all().slice(1).map(describe) : [],
    texts: view.children.filter(child => child.tagName !== "FIGURE").map(describe),
    all: elements.map(describe),
  };
}

const leaderboard = [{ model: "random_forest", title: "Random Forest", selected: true,
  cv_macro_f1_mean: 0.912, cv_macro_f1_std: 0.004, test_macro_f1: 0.944, fit_seconds: 12.5,
  test_balanced_accuracy: 0.95, test_accuracy: 0.99, test_min_recall: 0.8,
  test_recall_per_class: [["BENIGN", 0.99]] }];
const summary = { run_id: "20261103T120920Z-5a8b6862", created_at: "2026-11-03T12:09:20Z",
  dataset: { name: "d.csv", rows: 200 }, test_dataset: null, profile: null, label: "Label",
  split: "stratified", train_rows: 140, test_rows: 60, balancing: "none",
  models: ["random_forest"], selected: "random_forest", selected_title: "Random Forest",
  cv_macro_f1_mean: 0.912, test_macro_f1: 0.944 };
const view = { summary, leaderboard, folds: { random_forest: [0.91, 0.912, 0.914] } };
// a saved file's shape: its repeats and seed in settings, no saved flag
const { repeats, seed, ...file } = result({ concentrated: "feature 3" });
const saved = { ...file, settings: { max_rows: 5000, min_per_class: 3, repeats, seed, metric: "macro-F1" } };

const why = (extra = {}) => ({
  prediction: "DDoS", probability: 0.97, source: "training", problem: null, without_typical: [],
  toward: [{ feature: XSS, value: 1480, typical: 0, effect: 0.41 },
           { feature: "Flow IAT Max", value: 3128082.202, typical: 7698.2715, effect: 0.0204 }],
  away: [{ feature: "Idle Min", value: null, typical: 12, effect: -0.12 }], ...extra,
});
const read = view => ({ tag: view.tagName, open: "open" in describe(view).attributes,
  text: view.textContent,
  items: view.all().filter(item => item.tagName === "LI").map(item => item.textContent),
  all: view.all().map(describe) });

console.log(JSON.stringify({
  why: read(whyView(why())),
  why_none_toward: read(whyView(why({ toward: [], away: [] }))),
  why_old_bundle: read(whyView(why({ source: "pipeline", without_typical: ["proto", "service"] }))),
  why_no_probabilities: read(whyView(why({ problem: "NO_PROBABILITIES", probability: null, toward: [], away: [] }))),
  why_failed: read(whyView(null)),
  plain: shown(importanceView(result({ saved: true, not_saved: null }), "Random Forest")),
  concentrated: shown(importanceView(result({ concentrated: XSS }), "Random Forest")),
  not_saved: shown(importanceView(result({ saved: false, not_saved: "Not saved: only 10 MB free" }), "Random Forest")),
  few: shown(importanceView(result({ features: features.slice(0, 3) }), "Random Forest")),
  run_with: [runView(view, { importances: [saved], left_out: 0 })].flat(Infinity).filter(Boolean)
    .map(part => part.textContent).join(" | "),
  run_without: [runView(view, { importances: [], left_out: 1 })].flat(Infinity).filter(Boolean)
    .map(part => part.textContent).join(" | "),
}));
