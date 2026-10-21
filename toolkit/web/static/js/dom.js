// DOM helpers (S5). Text from the dataset, the server and the user (CSV headers, cell
// values, class names, error messages, predictions) is only ever set as text, never parsed
// as HTML: there is no innerHTML in the page's code, so a header like <img onerror=...>
// shows as those characters. Inline styles and on* handlers are refused as well: the CSP
// blocks them (S10), so they could only break the page.

const REFUSED = /^(on|style$|inner|outer|src)/i;  // handlers, inline styles, raw HTML, embedded pages

export function byId(id) {
  return document.getElementById(id);
}

// Append children (nested lists too): nodes as they are, anything else as text;
// null/undefined/false are skipped
function append(node, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) {
      continue;
    }
    node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

// el("td", { class: "x", title: "y" }, "text", otherNode, [more, children])
// Props: "class", "text", element properties (value, checked, htmlFor, type, ...) and
// other attributes (colspan, role, aria-*, ...).
export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (REFUSED.test(key)) {
      throw new Error("el(): '" + key + "' is not allowed");
    }
    if (value === null || value === undefined || value === false) {
      continue;
    }
    if (key === "class") {
      node.className = value;
    } else if (key === "text") {
      node.textContent = value;
    } else if (key in node) {
      node[key] = value;
    } else {
      node.setAttribute(key, value === true ? "" : value);
    }
  }
  return append(node, children);
}

// Replace a node's content with the given children (text or nodes)
export function setContent(node, ...children) {
  node.replaceChildren();
  return append(node, children);
}

export function show(node, visible = true) {
  node.hidden = !visible;
}

// One coloured status line (error, warning, success or info) as a node's only content
export function message(node, kind, text) {
  setContent(node, el("span", { class: "status-" + kind }, text));
}

// Fill a <select>: options are { value, text, selected }
export function fillSelect(select, options) {
  setContent(select, options.map(option =>
    el("option", { value: option.value, text: option.text, selected: Boolean(option.selected) })));
}

// Checkbox lists replace Ctrl+Click multi-selects, so they work on touch screens (U5).
export function fillCheckboxList(container, items) {
  setContent(container, items.map(item => el("label", { class: "checkbox-item" },
    el("input", { type: "checkbox", value: item.value, checked: Boolean(item.checked) }),
    // tooltip only: the checkbox keeps its visible text as its name
    el("span", { title: item.title || null }, item.text))));
}

export function checkedValues(container) {
  return Array.from(container.querySelectorAll("input[type=checkbox]:checked")).map(box => box.value);
}

export function setChecked(container, values) {
  container.querySelectorAll("input[type=checkbox]").forEach(box => {
    box.checked = values.includes(box.value);
  });
}

// A table with a header row: headers are texts, rows are arrays of cell texts or nodes
export function table(className, headers, rows) {
  return el("table", { class: className },
    el("thead", {}, el("tr", {}, headers.map(text => el("th", {}, text)))),
    el("tbody", {}, rows.map(cells => el("tr", {}, cells.map(cell => el("td", {}, cell))))));
}
