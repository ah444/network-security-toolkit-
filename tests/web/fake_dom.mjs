// A minimal DOM for running the page's modules in Node (tests/web/*.mjs): it never parses
// HTML, so text only becomes text nodes, as with the browser's createElement/append.

class FakeNode {}

class FakeText extends FakeNode {
  constructor(text) {
    super();
    this.data = String(text);
  }
  get textContent() {
    return this.data;
  }
}

export class FakeElement extends FakeNode {
  constructor(tag, namespace = null) {
    super();
    this.tagName = tag.toUpperCase();
    this.namespaceURI = namespace;
    this.childNodes = [];
    this.attributes = new Map();
    this.className = "";
    this.id = "";
    this.title = "";
    this.dir = "";
    this.hidden = false;
  }
  append(...children) {
    for (const child of children) {
      this.childNodes.push(child instanceof FakeNode ? child : new FakeText(child));
    }
  }
  replaceChildren(...children) {
    this.childNodes = [];
    this.append(...children);
  }
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }
  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }
  get textContent() {
    return this.childNodes.map(child => child.textContent).join("");
  }
  set textContent(value) {
    this.childNodes = [new FakeText(value)];
  }
  get children() {
    return this.childNodes.filter(child => child instanceof FakeElement);
  }
  all() {  // this element and every element under it, in document order
    return [this, ...this.children.flatMap(child => child.all())];
  }
}

// The page's modules make elements through these two, as in the browser
export function installFakeDom() {
  globalThis.Node = FakeNode;
  globalThis.document = {
    createElement: tag => new FakeElement(tag),
    createElementNS: (namespace, tag) => new FakeElement(tag, namespace),
  };
}
