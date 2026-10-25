"""Print, as JSON, where the toolkit finds its project folder, its data and its web files.

Run in a fresh process, so nothing is patched: by ``tests/test_paths.py`` from the project
folder, and after every commit of v5.2 step 2.3 with the Python of a scratch environment that
has the built wheel installed (``python -I tests/paths_probe.py``, outside the project), where
the toolkit must believe it is installed. Imports both the v5.1 and the v5.2 module names, so
the move cannot change what is expected.
"""

import json
from pathlib import Path

try:  # the v5.2 layout (step 2.3b and later)
    from toolkit.core.data import settings
except ModuleNotFoundError:  # the v5.1 layout
    from toolkit.data import settings
try:
    from toolkit.web.ui import STATIC, TEMPLATES
except ModuleNotFoundError:
    from toolkit.ui.web import STATIC, TEMPLATES

import toolkit

package = Path(toolkit.__file__).resolve().parent
print(json.dumps({
    "package": str(package),
    "repo_root": str(settings.REPO_ROOT),
    "is_checkout": settings.is_checkout(),
    "instance_dir": str(settings.instance_dir()),
    "env_file": str(settings.env_file()),
    "ENV_FILE": str(settings.ENV_FILE),
    "templates": sorted(p.name for p in TEMPLATES.glob("*.html")),
    "static": sorted(p.relative_to(STATIC).as_posix() for p in STATIC.rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts),
    "web_inside_package": all(package in path.resolve().parents for path in (TEMPLATES, STATIC)),
}))
