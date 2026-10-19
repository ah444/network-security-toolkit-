# UI before/after — v5.0 step 0.7

Screenshots of every section of the web app, taken in the built-in browser at a 1000×800 viewport, in light and dark mode. The trainer page was filled in with the same 90-row demo dataset each time (upload → class analysis → feature recommendation → training → confusion matrix → prediction form).

- `before/`: the original CSS (commit `b9fe304`)
- `after/`: after the step 0.7 fixes

Some "after" light-mode captures are cropped on the right. That's a scaling quirk of the screenshot tool at this pane size; the page itself has no horizontal overflow (measured: 985 px of content in a 1000 px viewport).

## What changed

| Problem (before) | Fix |
|---|---|
| **Light mode:** almost all text invisible (labels, inputs, preview table, results, metrics, home-page cards). A global `*:not(h1):not(h2) { color: white !important }` plus `.container`/`.section { color: white }` made it white on white (U1) | Rules removed; text inherits the page colour |
| **Light mode:** tables, messages, confusion matrix, slider and recommendation box styles missing. The dark-mode `@media` block in `style2.css` was never closed, so ~400 lines only applied in dark mode (U2) | Block closed after its last dark rule; the shared rules apply in both themes |
| **Dark mode:** Data Preview stat values invisible (`.stat-item { color: #2d3748 !important }`, the same colour as the background) | Inherit the theme colour |
| Confusion-matrix numbers: white on mid-green (3.6:1) | Lighter tints, always dark text (≈ 9:1) |
| "Incorrect" legend / hint text / section descriptions below 4.5:1 in one theme | Theme-specific colours (`#15803d`/`#b91c1c` light, `#4ade80`/`#fca5a5` dark) |
| `outline: none` on focused inputs, so keyboard users couldn't see focus | `:focus-visible` outline `#3730a3` (light) / `#fbbf24` (dark) |
| `'Inter'` font named but never loaded (U3) | System font stack, nothing downloaded |
| No viewport meta, no `lang` on the trainer page, home page loaded the trainer script (U4) | Added / removed |
| Prediction-form labels not linked to their inputs | `label.htmlFor` set |

## Contrast audit (WCAG AA)

`scripts/contrast_audit.js` checks every visible text element and form field against its effective background (4.5:1, or 3:1 for large text). Gradient backgrounds (navbar, buttons, recommendation box) are not measured; their text is white or light on a dark gradient.

| Page / theme | Elements checked | Below AA after the fix |
|---|---:|---:|
| Trainer, light | 202 | **0** |
| Trainer, dark | 202 | **0** |
| Home, light | 11 | **0** |

The "before" state wasn't measured with this script: light mode's text was invisible almost everywhere, as the screenshots show.

Keyboard check: with Tab, nav links and form fields show a solid 3 px focus outline in `#3730a3` (verified in the browser after the 0.3 s CSS transition settles).
