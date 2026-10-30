// Charts of one run (v5.3 F2d), drawn here as SVG: no library. Used by the trainer for the
// session's run and by the saved runs' page. Every number is also written as text beside its
// mark and in the chart's name for screen readers; colours come from the stylesheet's tokens
// through classes, so both themes work; text is set as text, never parsed (S5).

import { el, scrollBox } from "./dom.js";
import { fmt } from "./format.js";

const SVG = "http://www.w3.org/2000/svg";
const WIDTH = 600;  // drawn at its own size: a narrower screen scrolls its box
const NAME_END = 170;  // the model names end here (right-aligned)
const PLOT_START = 180;
const PLOT_END = 450;
const VALUE_START = 460;  // the numbers of each row
const ROW = 30;
const TOP = 34;  // room for a legend above the rows
const AXIS = 30;  // the tick labels under the rows

// An SVG element; its attributes come from this file only, its text is text
function svg(tag, attributes = {}, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attributes)) {
    node.setAttribute(key, String(value));
  }
  if (text !== undefined) {
    node.textContent = text;
  }
  return node;
}

// A chart: its title, the drawing (named by the description, which holds every number) in
// a box that scrolls on a narrow screen
function figure(title, description, rows, marks) {
  const height = TOP + rows * ROW + AXIS;
  const drawing = svg("svg", { class: "chart-drawing", width: WIDTH, height: height,
                               viewBox: `0 0 ${WIDTH} ${height}`, role: "img",
                               "aria-label": description });
  drawing.append(...marks.flat(Infinity).filter(Boolean));
  return el("figure", { class: "chart" }, el("figcaption", {}, title), scrollBox(title, drawing));
}

function rowMiddle(index) {
  return TOP + index * ROW + ROW / 2;
}

function modelName(index, title) {
  return svg("text", { class: "chart-label", x: NAME_END, y: rowMiddle(index) + 4,
                       "text-anchor": "end" }, title);
}

function rowValue(index, text) {
  return svg("text", { class: "chart-value", x: VALUE_START, y: rowMiddle(index) + 4 }, text);
}

// Where a score goes: the axis starts at the tenth below the lowest score (never below 0), so
// close scores stay apart; its start is written under it
function scoreScale(values) {
  let low = Math.max(0, Math.floor(Math.min(...values) * 10) / 10);
  if (low >= 1) {
    low = 0.9;
  }
  const at = value => PLOT_START + (Math.min(Math.max(value, low), 1) - low) / (1 - low) * (PLOT_END - PLOT_START);
  return { at, ticks: [low, (low + 1) / 2, 1], label: value => fmt(value, 2) };
}

// Seconds on a log scale, from a power of ten below the fastest to one above the slowest;
// at most 5 ticks (every second or third power when the times are far apart), so their
// labels never overlap
function secondsScale(values) {
  const positive = values.map(value => Math.max(value, 0.001));
  const low = Math.floor(Math.log10(Math.min(...positive)));
  const decades = Math.max(1, Math.ceil(Math.log10(Math.max(...positive))) - low);
  const step = Math.ceil(decades / 4);
  const high = low + step * Math.ceil(decades / step);
  const at = value => PLOT_START + (Math.log10(Math.max(value, 0.001)) - low) / (high - low) * (PLOT_END - PLOT_START);
  const ticks = [];
  for (let power = low; power <= high; power += step) {
    ticks.push(10 ** power);
  }
  return { at, ticks, label: seconds };
}

export function seconds(value) {
  return value >= 10 ? `${Math.round(value).toLocaleString("en-US")} s` :
    `${Number(value.toPrecision(2))} s`;
}

// The vertical grid lines and their labels under the rows
function axis(scale, rows) {
  const bottom = TOP + rows * ROW;
  return scale.ticks.map(tick => [
    svg("line", { class: "chart-grid", x1: scale.at(tick), x2: scale.at(tick), y1: TOP, y2: bottom }),
    svg("text", { class: "chart-tick", x: scale.at(tick), y: bottom + 18, "text-anchor": "middle" },
        scale.label(tick)),
  ]);
}

// One mark of the legend above the rows: its shape and its words
function legendItem(x, shape, words) {
  return [shape(x + 6, 14), svg("text", { class: "chart-tick", x: x + 16, y: 18 }, words)];
}

const dot = (x, y) => svg("circle", { class: "chart-cv", cx: x, cy: y, r: 5 });
const square = (x, y) => svg("rect", { class: "chart-test", x: x - 5, y: y - 5, width: 10, height: 10 });
const ring = (x, y) => svg("circle", { class: "chart-fold", cx: x, cy: y, r: 4 });

// Each model's CV macro-F1, fold by fold (rings), with its mean (the bar across) ± std (the line)
export function foldsChart(leaderboard, folds) {
  const values = leaderboard.flatMap(row => [...(folds[row.model] || []),
                                             row.cv_macro_f1_mean - row.cv_macro_f1_std]);
  const scale = scoreScale(values);
  const marks = leaderboard.map((row, i) => {
    const y = rowMiddle(i);
    const mean = row.cv_macro_f1_mean;
    const spread = row.cv_macro_f1_std;
    return [
      modelName(i, row.title),
      svg("line", { class: "chart-spread", x1: scale.at(mean - spread), x2: scale.at(mean + spread), y1: y, y2: y }),
      (folds[row.model] || []).map(score => ring(scale.at(score), y)),
      svg("line", { class: "chart-mean", x1: scale.at(mean), x2: scale.at(mean), y1: y - 9, y2: y + 9 }),
      rowValue(i, `${fmt(mean, 3)} ± ${fmt(spread, 3)}`),
    ];
  });
  const legend = [legendItem(PLOT_START, ring, "a fold"),
                  svg("line", { class: "chart-mean", x1: PLOT_START + 86, x2: PLOT_START + 86, y1: 6, y2: 22 }),
                  svg("text", { class: "chart-tick", x: PLOT_START + 96, y: 18 }, "the mean ± std")];
  const foldsText = row => folds[row.model] && folds[row.model].length ?
    `folds ${folds[row.model].map(score => fmt(score, 3)).join(", ")}; ` : "";
  const description = "CV macro-F1 on the training split, fold by fold. " + leaderboard.map(row =>
    `${row.title}: ${foldsText(row)}mean ${fmt(row.cv_macro_f1_mean, 3)} ± ` +
    fmt(row.cv_macro_f1_std, 3)).join(". ") + ".";
  return figure("CV macro-F1, fold by fold", description, leaderboard.length,
                [legend, axis(scale, leaderboard.length), marks]);
}

// Each model's CV macro-F1 (circle) against its test macro-F1 (square): a big gap means the
// CV estimate didn't hold on the test rows
export function cvTestChart(leaderboard) {
  const scale = scoreScale(leaderboard.flatMap(row => [row.cv_macro_f1_mean, row.test_macro_f1]));
  const marks = leaderboard.map((row, i) => {
    const y = rowMiddle(i);
    const cv = scale.at(row.cv_macro_f1_mean);
    const test = scale.at(row.test_macro_f1);
    return [
      modelName(i, row.title),
      svg("line", { class: "chart-spread", x1: Math.min(cv, test), x2: Math.max(cv, test), y1: y, y2: y }),
      dot(cv, y), square(test, y),
      rowValue(i, `CV ${fmt(row.cv_macro_f1_mean, 3)}, test ${fmt(row.test_macro_f1, 3)}`),
    ];
  });
  const legend = [legendItem(PLOT_START, dot, "CV (training split)"),
                  legendItem(PLOT_START + 150, square, "test (scored once)")];
  const description = "CV macro-F1 against test macro-F1. " + leaderboard.map(row =>
    `${row.title}: CV ${fmt(row.cv_macro_f1_mean, 3)}, test ${fmt(row.test_macro_f1, 3)}`).join(". ") + ".";
  return figure("CV against test macro-F1", description, leaderboard.length,
                [legend, axis(scale, leaderboard.length), marks]);
}

// The time to fit each model on the whole training split, on a log scale
export function timeChart(leaderboard) {
  const scale = secondsScale(leaderboard.map(row => row.fit_seconds));
  const marks = leaderboard.map((row, i) => {
    const y = rowMiddle(i);
    return [
      modelName(i, row.title),
      svg("line", { class: "chart-spread", x1: PLOT_START, x2: scale.at(row.fit_seconds), y1: y, y2: y }),
      dot(scale.at(row.fit_seconds), y),
      rowValue(i, seconds(row.fit_seconds)),
    ];
  });
  const description = "Fit time on the training split, log scale. " + leaderboard.map(row =>
    `${row.title}: ${seconds(row.fit_seconds)}`).join(". ") + ".";
  return figure("Fit time (log scale)", description, leaderboard.length,
                [axis(scale, leaderboard.length), marks]);
}

// The shade of a recall: none from 0.99 up, darker the lower (the confusion matrix's steps,
// whose dark text stays >= 4.5:1)
export function recallShade(recall) {
  const steps = [0.99, 0.95, 0.9, 0.8, 0.5];
  const step = steps.filter(limit => recall < limit).length;
  return step === 0 ? null : `cm-wrong-${step}`;
}

// The test recall of each class (rows) for each model (columns), every value written
export function recallTable(leaderboard) {
  const classes = [];
  leaderboard.forEach(row => row.test_recall_per_class.forEach(([label]) => {
    if (!classes.includes(label)) {
      classes.push(label);
    }
  }));
  const recalls = leaderboard.map(row => new Map(row.test_recall_per_class));
  const title = "Test recall per class";
  return el("figure", { class: "chart" }, el("figcaption", {}, title),
    el("p", { class: "chart-note" }, "Shaded: under 0.99, darker the lower (the darkest under 0.5)."),
    scrollBox(title, el("table", { class: "confusion-matrix recall-table" },
      el("thead", {}, el("tr", {}, el("th", { scope: "col" }, "Class ↓ / Model →"),
        leaderboard.map(row => el("th", { scope: "col" }, row.title)))),
      el("tbody", {}, classes.map(label => el("tr", {}, el("th", { scope: "row" }, label),
        recalls.map(recall => recall.has(label) ?
          el("td", { class: recallShade(recall.get(label)) }, fmt(recall.get(label), 3)) :
          el("td", {}, "–"))))))));
}

// All the charts of a run, in a section that can be closed
export function runCharts(leaderboard, folds) {
  return el("details", { class: "run-charts", open: true },
    el("summary", {}, "Charts"),
    foldsChart(leaderboard, folds || {}), cvTestChart(leaderboard), recallTable(leaderboard),
    timeChart(leaderboard));
}
