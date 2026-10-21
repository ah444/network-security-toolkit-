// Page state shared by the modules, and the events they use to tell each other what
// happened (instead of calling each other):
//   dataset:loaded  {columns, dataset}  a dataset was uploaded
//   dataset:changed                     classes were removed from it
//   run:ready       result              a training run finished
//   bundle:loaded   result              an exported bundle was loaded

import { byId } from "./dom.js";

export const state = {
  modelTitles: {},  // model id -> title, from /api/v1/models
  confusion: {},    // test-split confusion matrices of the last run: model -> {labels, matrix}
  target: null,     // the target column analysed for the class filter
};

export function modelTitle(id) {
  return state.modelTitles[id] || id;
}

// Test split size from the slider (0.1 - 0.5)
export function testSize() {
  return parseInt(byId("trainTestSlider").value) / 100;
}

export function emit(name, detail) {
  document.dispatchEvent(new CustomEvent(name, { detail: detail }));
}

export function on(name, handler) {
  document.addEventListener(name, event => handler(event.detail));
}
