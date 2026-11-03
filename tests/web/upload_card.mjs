// Runs the page's real modules (summary.js, data.js, format.js) in Node with a minimal DOM,
// and prints what they build as JSON for tests/web/test_upload_card.py. The DOM here never
// parses HTML: text only becomes text nodes, as with the browser's createElement/append,
// so a name that reached the page as HTML would be missing from the text below.
// Usage: node upload_card.mjs <the js folder>

import { pathToFileURL } from "node:url";
import path from "node:path";

import { installFakeDom } from "./fake_dom.mjs";

installFakeDom();

const folder = process.argv[2];
const load = name => import(pathToFileURL(path.join(folder, name)).href);
const { datasetSummary, uploadError } = await load("data.js");
const { chips } = await load("summary.js");
const { wholeNumber } = await load("format.js");

// What a test needs to see of a panel: its attributes, heading, tiles and footnote
function panel(node) {
  const elements = node.all();
  const heading = elements.find(item => item.tagName === "H4");
  return {
    tag: node.tagName,
    class: node.className,
    role: node.getAttribute("role"),
    labelledby: node.getAttribute("aria-labelledby"),
    heading_id: heading.id,
    heading: heading.textContent,
    icon: heading.children[0].tagName,
    icon_hidden: heading.children[0].getAttribute("aria-hidden"),
    list: elements.filter(item => item.tagName === "DL").length,
    tiles: elements.filter(item => item.className === "stat-tile").map(tile => ({
      label: tile.children[0].textContent,
      value: tile.children[1].textContent,
      classes: tile.children[1].all().map(item => item.className).filter(Boolean),
    })),
    paragraphs: elements.filter(item => item.tagName === "P").map(item => item.textContent),
    tags: [...new Set(elements.map(item => item.tagName))].sort(),
  };
}

function chipList(names) {
  return chips(names).flat().filter(Boolean).map(chip => ({
    text: chip.textContent, class: chip.className, title: chip.title, dir: chip.dir,
  }));
}

const XSS = "<img src=x onerror=alert(1)>.csv";
const nsl = {
  files: ["KDDTest+.txt", "KDDTrain+.txt"], rows: 125973, memory_mb: 23.1,
  test_file: "KDDTest+.txt", test_rows: 22544,
  profile: { name: "nsl_kdd", variant: "full", label_column: "attack",
             split_strategy: "official", blocked_columns: ["level"], warnings: [] },
};
const single = { ...nsl, files: ["flows.csv"], rows: 2000, memory_mb: 0.6,
                 test_file: null, test_rows: null,
                 profile: { ...nsl.profile, name: "cic_ids2017", variant: "original",
                            label_column: "Label", split_strategy: "stratified",
                            blocked_columns: [] } };
const many = { ...single, profile: { ...single.profile, split_strategy: "temporal_per_class",
                                     blocked_columns: ["Flow ID", "Source IP", "Timestamp"] } };
const unknown = { ...single, profile: { ...single.profile, name: "generic", variant: "generic",
                                        label_column: null } };
const hostile = { ...nsl, files: [XSS, "b.csv"], test_file: XSS,
                  profile: { ...nsl.profile, name: XSS, variant: XSS, label_column: XSS,
                             blocked_columns: [XSS] } };
const names = count => Array.from({ length: count }, (_, i) => `part-${i + 1}.csv`);

console.log(JSON.stringify({
  nsl: panel(datasetSummary(nsl)),
  single: panel(datasetSummary(single)),
  many: panel(datasetSummary(many)),
  unknown: panel(datasetSummary(unknown)),
  hostile: panel(datasetSummary(hostile)),
  error: panel(uploadError("Could not read " + XSS + " as a CSV/TXT dataset.")),
  chips_none: chipList([]),
  chips_six: chipList(names(6)),
  chips_seven: chipList(names(7)),
  chips_twenty: chipList(names(20)),
  chips_hostile: chipList([XSS, "تدريب.csv"]),
  numbers: [wholeNumber(125973), wholeNumber(0), wholeNumber(2)],
}));
