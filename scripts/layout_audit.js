// Layout audit (plan D2, v5.2 step 2.2). Paste into the browser console on a page of the
// web app, or run it through a browser automation tool, at each screen size to check
// (step 2.2 used 320 x 568, 375 x 812, 414 x 896, 667 x 375, 768 x 1024, 1024 x 768,
// 1280 x 800 and 1920 x 1080). Returns what makes a page hard to use on that screen:
// sideways scroll of the whole page, elements cut off by the screen edge (not inside a
// box that scrolls), tap targets under 44 px, checkbox and radio rows under 44 px, text
// under 12 px, how much of the screen the sticky parts cover, the content width of the
// first card, the step bar's height and the page's length in screens. Empty lists are left
// out. Boxes that scroll sideways (wide tables) are listed: that is by design, not a fault.
(() => {
  const W = document.documentElement.clientWidth; // without a vertical scrollbar
  const H = innerHeight;
  const name = (el) =>
    el.tagName.toLowerCase() +
    (el.id ? "#" + el.id : el.classList[0] ? "." + el.classList[0] : "") +
    (el.type && ["checkbox", "radio", "number", "file", "submit", "button", "text"].includes(el.type)
      ? "[" + el.type + "]"
      : "");
  const shown = (el) => {
    if (!el.getClientRects().length) return false;
    const style = getComputedStyle(el);
    return style.visibility !== "hidden" && style.display !== "none";
  };
  const scrolls = (el) => ["auto", "scroll"].includes(getComputedStyle(el).overflowX);
  const inScroller = (el, clipped) => {
    for (let node = el.parentElement; node && node !== document.documentElement; node = node.parentElement) {
      const overflow = getComputedStyle(node).overflowX;
      if (overflow === "auto" || overflow === "scroll" || (clipped && overflow === "hidden")) return true;
    }
    return false;
  };
  // The most frequent entries first, "x3" when one kind appears more than once
  const top = (counts) =>
    [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .map(([key, count]) => key + (count > 1 ? " x" + count : ""));
  const all = [...document.querySelectorAll("body *")].filter(shown);

  const outside = new Map();
  for (const el of all) {
    const box = el.getBoundingClientRect();
    if ((box.right > W + 1 || box.left < -1) && !inScroller(el, true)) {
      outside.set(name(el), Math.max(outside.get(name(el)) || 0, Math.round(box.right - W)));
    }
  }

  const innerScroll = all
    .filter((el) => scrolls(el) && el.scrollWidth > el.clientWidth + 1)
    .map((el) => name(el) + ": " + el.clientWidth + " of " + el.scrollWidth + " px");

  const small = new Map();
  const targets = "a[href], button, select, textarea, input:not([type=hidden]):not([type=checkbox]):not([type=radio])";
  for (const el of all.filter((el) => el.matches(targets))) {
    const box = el.getBoundingClientRect();
    if (box.height < 43.5 || box.width < 43.5) {
      const key = name(el) + " " + Math.round(box.width) + "x" + Math.round(box.height);
      small.set(key, (small.get(key) || 0) + 1);
    }
  }

  const boxes = all.filter((el) => el.matches("input[type=checkbox], input[type=radio]"));
  const smallRows = boxes.filter((el) => (el.closest("label") || el).getBoundingClientRect().height < 43.5);

  const tiny = new Map();
  for (const el of all) {
    const ownText = [...el.childNodes].some((node) => node.nodeType === 3 && node.textContent.trim());
    const size = parseFloat(getComputedStyle(el).fontSize);
    if (ownText && size < 12) {
      const key = name(el) + " " + size + "px";
      tiny.set(key, (tiny.get(key) || 0) + 1);
    }
  }

  // Sticky headers of a table that scrolls inside its own box don't cover the page
  const sticky = all.filter(
    (el) => ["sticky", "fixed"].includes(getComputedStyle(el).position) && !inScroller(el, false)
  );
  const stickyHeight = sticky.reduce((sum, el) => sum + el.getBoundingClientRect().height, 0);

  const result = {
    size: innerWidth + " x " + H,
    pageScrollX: document.documentElement.scrollWidth - W,
    outOfScreen: [...outside.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6).map(([key, px]) => key + " +" + px + "px"),
    innerScroll: innerScroll.slice(0, 6),
    smallTargets: top(small),
    smallCheckboxRows: smallRows.length + " of " + boxes.length,
    tinyText: top(tiny),
    sticky: sticky.map((el) => name(el) + " " + Math.round(el.getBoundingClientRect().height) + "px"),
    stickyShare: Math.round((stickyHeight / H) * 100) + "%",
    screens: Math.round((document.documentElement.scrollHeight / H) * 10) / 10,
  };
  const card = all.find((el) => el.matches(".section"));
  if (card) {
    const style = getComputedStyle(card);
    const content = card.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
    result.sectionContent = Math.round(content) + " of " + innerWidth + " px";
  }
  const stepper = document.querySelector(".stepper");
  if (stepper) {
    result.stepper = Math.round(stepper.getBoundingClientRect().height) + "px";
    const current = stepper.querySelector("[aria-current]");
    if (current) {
      const bar = stepper.querySelector(".stepper-list").getBoundingClientRect();
      const box = current.getBoundingClientRect();
      result.currentStepInView = box.left >= bar.left - 1 && box.right <= bar.right + 1;
    }
  }
  for (const key of Object.keys(result)) {
    if (Array.isArray(result[key]) && !result[key].length) delete result[key];
  }
  return result;
})();
