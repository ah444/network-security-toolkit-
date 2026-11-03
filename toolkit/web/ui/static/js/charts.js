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
                  legendItem(PLOT_START + 150, square, "test split")];
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

// The leaderboard of a run, ranked by CV macro-F1, the selected model starred (the trainer's
// run and a saved one)
export function leaderboardTable(leaderboard) {
  return scrollBox("Leaderboard", el("table", { class: "metrics-table" },
    el("thead", {}, el("tr", {}, ["Model", "CV macro-F1", "Test macro-F1", "Test balanced acc.",
                                  "Test accuracy", "Weakest class recall", "Fit time"].map(text => el("th", {}, text)))),
    el("tbody", {}, leaderboard.map(row => el("tr", {},
      [(row.selected ? "★ " : "") + row.title,
       `${fmt(row.cv_macro_f1_mean, 3)} ± ${fmt(row.cv_macro_f1_std, 3)}`,
       fmt(row.test_macro_f1, 3), fmt(row.test_balanced_accuracy, 3), fmt(row.test_accuracy, 3),
       fmt(row.test_min_recall, 3), `${fmt(row.fit_seconds, 1)} s`].map(text => el("td", {}, text)))))));
}

// All the charts of a run, in a section that can be closed
export function runCharts(leaderboard, folds) {
  return el("details", { class: "run-charts", open: true },
    el("summary", {}, "Charts"),
    foldsChart(leaderboard, folds || {}), cvTestChart(leaderboard), recallTable(leaderboard),
    timeChart(leaderboard));
}

// --- What a model relies on (v5.3 F3e) ---------------------------------------------------------

const TOP_FEATURES = 15;  // the features with the biggest drop that the chart shows
const NAME_CHARS = 28;  // a longer name is cut, so it never runs into the bars (its whole
                        // name is in the drawing's name)
const drop = value => value.toFixed(4);

// The drop in macro-F1 when each feature's values are shuffled (the bar: its mean, from 0; the
// line: ± its std over the repeats); a negative drop (shuffling helped) goes left of 0
function importanceChart(features, title) {
  const low = Math.min(0, ...features.map(f => f.mean - f.std));
  const high = Math.max(...features.map(f => f.mean + f.std), low + 0.001);
  const at = value => PLOT_START + (value - low) / (high - low) * (PLOT_END - PLOT_START);
  const scale = { at, ticks: low < 0 ? [low, 0, high] : [0, high / 2, high],
                  label: value => fmt(value, 3) };
  const marks = features.map((f, i) => {
    const y = rowMiddle(i);
    const name = f.feature.length > NAME_CHARS ? f.feature.slice(0, NAME_CHARS - 1).trimEnd() + "…" : f.feature;
    return [
      modelName(i, name),
      svg("rect", { class: "chart-drop", x: Math.min(at(0), at(f.mean)), y: y - 7,
                    width: Math.abs(at(f.mean) - at(0)), height: 14 }),
      svg("line", { class: "chart-spread", x1: at(f.mean - f.std), x2: at(f.mean + f.std), y1: y, y2: y }),
      rowValue(i, `${drop(f.mean)} ± ${drop(f.std)}`),
    ];
  });
  const bottom = TOP + features.length * ROW;
  const legend = svg("text", { class: "chart-tick", x: PLOT_START, y: 18 },
                     "bar: the mean drop; line: ± std over the repeats");
  const zero = svg("line", { class: "chart-zero", x1: at(0), x2: at(0), y1: TOP, y2: bottom });
  const description = "The drop in macro-F1 when each feature's values are shuffled, mean ± std " +
    "over the repeats. " + features.map(f => `${f.feature}: ${drop(f.mean)} ± ${drop(f.std)}`).join(". ") + ".";
  return figure(title, description, features.length, [legend, axis(scale, features.length), zero, marks]);
}

// One importance result (a job's, or a saved file's: its repeats and seed in ``settings``):
// the chart of the features with the biggest drop, what the numbers are, the concentration
// warning (HC2), and why it was not saved
export function importanceView(result, title) {
  const shown = result.features.slice(0, TOP_FEATURES);
  const settings = result.settings || result;
  const count = number => number.toLocaleString("en-US");
  return el("div", { class: "importance-result" },
    importanceChart(shown, `What ${title} relies on`),
    el("p", { class: "chart-note" },
       `Scored on ${count(result.rows)} of ${count(result.test_rows)} test rows, ${settings.repeats} repeats, seed ${settings.seed}; ` +
       `the ${shown.length} of ${result.features.length} features with the biggest drop are shown.`),
    el("p", { class: "chart-note" },
       "A diagnosis of this run on its test rows, not a way to choose features or models; " +
       "it shows what the model relies on, not causes."),
    result.concentrated && el("div", { class: "warning-message" },
      `⚠️ ${result.concentrated} carries more than half of the effect. Check that it is not a ` +
      "copy of the label (one possible cause) before trusting the scores."),
    result.saved === false && el("div", { class: "warning-message" }, result.not_saved));
}
