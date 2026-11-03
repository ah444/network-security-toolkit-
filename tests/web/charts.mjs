// Runs the page's charts.js in Node with the minimal DOM of fake_dom.mjs, and prints what it
// draws as JSON for tests/web/test_charts.py.
// Usage: node charts.mjs <the js folder>

import { pathToFileURL } from "node:url";
import path from "node:path";

import { installFakeDom } from "./fake_dom.mjs";

installFakeDom();

const folder = process.argv[2];
const { runCharts, recallShade, seconds } = await import(pathToFileURL(path.join(folder, "charts.js")).href);

const XSS = "<img src=x onerror=alert(1)>";
const row = (model, title, cv, std, test, fit, recalls) => ({
  model, title, selected: model === "random_forest", cv_macro_f1_mean: cv, cv_macro_f1_std: std,
  test_macro_f1: test, fit_seconds: fit, test_recall_per_class: recalls,
});
const leaderboard = [
  row("random_forest", "Random Forest", 0.912, 0.004, 0.944, 4414.07,
      [["BENIGN", 0.9937], ["DoS", 0.85], [XSS, 0.3]]),
  row("decision_tree", "Decision Tree", 0.881, 0.012, 0.902, 12.5,
      [["BENIGN", 0.991], ["DoS", 0.96], [XSS, 0.62]]),
  row("naive_bayes", XSS, 0.55, 0.03, 0.4, 0.0042,
      [["BENIGN", 0.7], ["DoS", 0.92], [XSS, 0.0]]),
];
const folds = { random_forest: [0.908, 0.913, 0.915], decision_tree: [0.87, 0.88, 0.893],
                naive_bayes: [0.52, 0.55, 0.58] };

// What a test needs of an element: its tag, class, attributes and text
function describe(node) {
  const attributes = Object.fromEntries(node.attributes || []);
  return { tag: node.tagName, class: node.className || attributes.class || "",
           attributes, text: node.textContent };
}

function chart(figure) {
  const elements = figure.all();
  const drawing = elements.find(item => item.tagName === "SVG");
  return {
    caption: elements.find(item => item.tagName === "FIGCAPTION").textContent,
    box: describe(elements.find(item => item.className === "table-container")),
    drawing: drawing ? describe(drawing) : null,
    marks: drawing ? drawing.all().slice(1).map(describe) : [],
    cells: elements.filter(item => ["TH", "TD"].includes(item.tagName)).map(describe),
    note: (elements.find(item => item.tagName === "P") || { textContent: null }).textContent,
  };
}

const charts = runCharts(leaderboard, folds);
const children = charts.children;
console.log(JSON.stringify({
  section: describe(charts),
  summary: children[0].textContent,
  figures: children.slice(1).map(chart),
  without_folds: runCharts(leaderboard, undefined).children.slice(1).map(chart)[0].drawing.attributes["aria-label"],
  one_model: chart(runCharts([leaderboard[0]], folds).children[1]).drawing.attributes,
  same_speed: chart(runCharts([leaderboard[1], { ...leaderboard[1], model: "x" }], folds).children[4]).marks
    .filter(mark => mark.class === "chart-tick").map(mark => mark.text),
  shades: [1, 0.99, 0.985, 0.95, 0.93, 0.9, 0.85, 0.8, 0.6, 0.5, 0.2, 0].map(recallShade),
  seconds: [0.0042, 0.5, 1, 9.99, 12.5, 4414.07].map(seconds),
}));
