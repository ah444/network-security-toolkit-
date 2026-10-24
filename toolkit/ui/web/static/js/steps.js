// Steps (v5.1 step 1.3c): one step of the trainer is shown at a time. Explore to Train
// unlock when a dataset is loaded, Use when a run is ready or a bundle is loaded; the
// bundle path has Data and Use only. The last step viewed is kept for this tab, so a
// reload opens it again.

import { byId, show } from "./dom.js";
import { on } from "./state.js";

const STEPS = ["data", "explore", "clean", "features", "train", "use"];
const DATASET_STEPS = ["explore", "clean", "features", "train"];
const STORAGE_KEY = "ids-trainer-step";

const unlocked = new Set(["data"]);
let path = "upload";
let current = "data";

function pathSteps() {
  return path === "bundle" ? ["data", "use"] : STEPS;
}

function neighbour(step, offset) {
  const steps = pathSteps();
  return steps[steps.indexOf(step) + offset];
}

// Step bar, step sections and Next buttons follow `current` and `unlocked`
function refresh() {
  const steps = pathSteps();
  document.querySelectorAll(".step-link").forEach(link => {
    const step = link.dataset.step;
    show(link.parentNode, steps.includes(step));
    link.disabled = !unlocked.has(step);
    if (step === current) {
      link.setAttribute("aria-current", "step");
    } else {
      link.removeAttribute("aria-current");
    }
  });
  document.querySelectorAll("section.step").forEach(section => {
    const step = section.dataset.step;
    show(section, step === current);
    const next = section.querySelector(".step-next");
    if (next) {
      next.disabled = !unlocked.has(neighbour(step, 1));
    }
  });
}

function remember(step) {
  try {
    sessionStorage.setItem(STORAGE_KEY, step);
  } catch (error) {
    // storage blocked (private window, site data off): a reload opens the furthest step
  }
}

function recalled() {
  try {
    return sessionStorage.getItem(STORAGE_KEY);
  } catch (error) {
    return null;
  }
}

// Show a step; after a click the keyboard focus moves to its heading
function go(step, focus = true) {
  if (!unlocked.has(step) || !pathSteps().includes(step)) {
    return;
  }
  current = step;
  refresh();
  remember(step);
  if (focus) {
    byId("step-" + step + "-title").focus();
  }
}

function unlock(steps) {
  steps.forEach(step => unlocked.add(step));
  refresh();
}

// After the page has resumed: the last step viewed, else the furthest one unlocked
function restore() {
  const saved = recalled();
  const furthest = pathSteps().filter(step => unlocked.has(step)).pop();
  go(saved && unlocked.has(saved) && pathSteps().includes(saved) ? saved : furthest, false);
}

export function setup() {
  document.querySelectorAll(".step-link").forEach(link => {
    link.addEventListener("click", () => go(link.dataset.step));
  });
  document.querySelectorAll("section.step").forEach(section => {
    const step = section.dataset.step;
    section.querySelectorAll(".step-next").forEach(button => {
      button.addEventListener("click", () => go(neighbour(step, 1)));
    });
    section.querySelectorAll(".step-back").forEach(button => {
      button.addEventListener("click", () => go(neighbour(step, -1)));
    });
  });
  on("path:changed", newPath => {
    path = newPath;
    unlocked.clear();
    unlocked.add("data");
    current = "data";
    refresh();
  });
  on("dataset:loaded", () => unlock(DATASET_STEPS));
  on("run:ready", () => unlock(["use"]));
  on("bundle:loaded", () => unlock(["use"]));
  on("steps:restore", restore);
  refresh();
}
