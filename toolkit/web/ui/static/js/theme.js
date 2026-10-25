// Theme (v5.2 step 2.1): Auto follows the system's light or dark setting, Light and Dark
// override it, and the choice is kept in this browser. A classic script in the <head> of
// both pages, not a module: it runs before the page is drawn, so the page never shows the
// other theme first. Without JavaScript the button stays hidden and the page follows the
// system. Light and Dark set data-theme on <html>; app.css does the rest.

(() => {
  const STORAGE_KEY = "toolkit-theme";
  const CHOICES = ["auto", "light", "dark"];
  const LABELS = { auto: "Auto", light: "Light", dark: "Dark" };
  const root = document.documentElement;

  function saved() {
    try {
      const value = localStorage.getItem(STORAGE_KEY);
      return CHOICES.includes(value) ? value : "auto";
    } catch (error) {
      return "auto"; // storage blocked (private window, site data off)
    }
  }

  let choice = saved();

  function save() {
    try {
      localStorage.setItem(STORAGE_KEY, choice);
    } catch (error) {
      // storage blocked: the choice holds until the page is left
    }
  }

  // The attribute for the CSS, and the button's text for whoever reads or hears it
  function apply() {
    if (choice === "auto") {
      root.removeAttribute("data-theme");
    } else {
      root.setAttribute("data-theme", choice);
    }
    const button = document.getElementById("themeToggle");
    if (button) {
      button.textContent = `Theme: ${LABELS[choice]}`;
      const detail = choice === "auto" ? " (follows the system)" : "";
      button.setAttribute("aria-label", `Theme: ${LABELS[choice]}${detail}. Change theme`);
    }
  }

  apply();

  document.addEventListener("DOMContentLoaded", () => {
    const button = document.getElementById("themeToggle");
    if (!button) {
      return;
    }
    button.addEventListener("click", () => {
      choice = CHOICES[(CHOICES.indexOf(choice) + 1) % CHOICES.length];
      save();
      apply();
    });
    apply();
    button.hidden = false;
  });
})();
