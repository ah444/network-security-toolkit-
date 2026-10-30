// Navbar menu (v5.2 step 2.2a): on screens 768 px wide and less the navbar is one row, the
// logo and a Menu button that opens the links and the Theme button. A classic script right
// after the navbar, so the menu is closed before the page is drawn. Without JavaScript the
// button stays hidden and the links show under the logo. app.css hides a closed menu on
// small screens only; on wider ones the links always show and the button is not drawn.

(() => {
  const nav = document.querySelector(".navbar");
  const button = document.getElementById("navToggle");
  if (!nav || !button) {
    return;
  }

  function setOpen(open) {
    nav.setAttribute("data-menu", open ? "open" : "closed");
    button.setAttribute("aria-expanded", String(open));
  }

  button.addEventListener("click", () => setOpen(button.getAttribute("aria-expanded") !== "true"));
  // Escape inside the navbar closes the menu; the keyboard focus goes back to the button
  nav.addEventListener("keydown", event => {
    if (event.key === "Escape" && button.getAttribute("aria-expanded") === "true") {
      setOpen(false);
      button.focus();
    }
  });
  nav.querySelectorAll(".nav-menu a").forEach(link => {
    link.addEventListener("click", () => setOpen(false));
  });
  setOpen(false);
  button.hidden = false;
})();
