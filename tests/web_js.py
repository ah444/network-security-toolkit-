"""Test helpers for the web page's ES modules (v5.1 step 1.3a): read them, without comments."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JS_DIR = REPO_ROOT / "toolkit/web/ui/static/js"
TEMPLATE = REPO_ROOT / "toolkit/web/ui/templates/dynamic_ids.html"


def found(paths, what):
    """``paths`` sorted; an error when there are none, so a folder that moved (v5.2 step 2.3)
    makes the tests fail instead of passing over an empty list."""
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"no {what} found: did the folder move?")
    return paths


TEMPLATES = found(TEMPLATE.parent.glob("*.html"), "templates")


def _without_comments(text):
    """Drop // line comments (outside strings) and /* */ blocks, so rules check code only."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    lines = []
    for line in text.splitlines():
        quote, cut = None, len(line)
        for i, ch in enumerate(line):
            if quote:
                if ch == "\\":
                    continue
                if ch == quote:
                    quote = None
            elif ch in "'\"`":
                quote = ch
            elif line.startswith("//", i):
                cut = i
                break
        lines.append(line[:cut])
    return "\n".join(lines)


def js_sources(code_only=True):
    """Module file name -> its text (comments removed unless ``code_only`` is False)."""
    texts = {path.name: path.read_text(encoding="utf-8")
             for path in found(JS_DIR.glob("*.js"), "JavaScript modules")}
    return {name: _without_comments(text) for name, text in texts.items()} if code_only else texts


def all_js(code_only=True):
    return "\n".join(js_sources(code_only).values())
