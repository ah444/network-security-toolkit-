// Contrast audit (plan D2, step 0.7). Paste into the browser console on a page of the
// web app, or run it through a browser automation tool. Returns every visible text
// element whose text/background contrast is below WCAG AA (4.5:1, or 3:1 for large
// text). Backgrounds with gradients (buttons, navbar, recommendation box) are reported
// separately as "unmeasured" because a single ratio can't describe them.
(() => {
  const parse = (c) => {
    const m = c.match(/rgba?\(([^)]+)\)/);
    if (!m) return null;
    const [r, g, b, a = 1] = m[1].split(",").map((v) => parseFloat(v));
    return { r, g, b, a };
  };
  const lum = ({ r, g, b }) => {
    const ch = [r, g, b].map((v) => {
      v /= 255;
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2];
  };
  const blend = (top, bottom) => ({
    r: top.r * top.a + bottom.r * (1 - top.a),
    g: top.g * top.a + bottom.g * (1 - top.a),
    b: top.b * top.a + bottom.b * (1 - top.a),
    a: 1,
  });
  // Effective background: blend translucent layers down to the first opaque one.
  const background = (el) => {
    const layers = [];
    for (let node = el; node; node = node.parentElement) {
      const style = getComputedStyle(node);
      if (style.backgroundImage && style.backgroundImage !== "none") return "gradient";
      const bg = parse(style.backgroundColor);
      if (bg && bg.a > 0) {
        layers.push(bg);
        if (bg.a >= 1) break;
      }
    }
    let result = { r: 255, g: 255, b: 255, a: 1 };
    for (const layer of layers.reverse()) result = blend(layer, result);
    return result;
  };
  const failures = [];
  const unmeasured = new Set();
  let checked = 0;
  for (const el of document.querySelectorAll("body *")) {
    const ownText = [...el.childNodes]
      .filter((n) => n.nodeType === 3)
      .map((n) => n.textContent.trim())
      .join(" ")
      .trim();
    const isField = ["INPUT", "SELECT", "TEXTAREA"].includes(el.tagName)
      && !["file", "radio", "checkbox", "range"].includes(el.type);
    if (!ownText && !isField) continue;
    if (!el.offsetParent && getComputedStyle(el).position !== "fixed") continue;
    const style = getComputedStyle(el);
    if (style.visibility === "hidden" || parseFloat(style.opacity) === 0) continue;
    const bg = background(el);
    const label = (ownText || el.value || el.id || el.tagName).slice(0, 40);
    if (bg === "gradient") {
      unmeasured.add(label);
      continue;
    }
    const fg = blend(parse(style.color), bg);
    const [l1, l2] = [lum(fg), lum(bg)].sort((a, b) => b - a);
    const ratio = (l1 + 0.05) / (l2 + 0.05);
    const size = parseFloat(style.fontSize);
    const bold = parseInt(style.fontWeight, 10) >= 700;
    const needed = size >= 24 || (bold && size >= 18.66) ? 3 : 4.5;
    checked += 1;
    if (ratio < needed) {
      failures.push({ text: label, ratio: Math.round(ratio * 100) / 100, needed, color: style.color });
    }
  }
  return { checked, failures, unmeasuredGradientTexts: [...unmeasured].slice(0, 12) };
})();
