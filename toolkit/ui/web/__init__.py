"""The web trainer's page: ``templates/`` and ``static/`` (one CSS file, ES modules;
``static/js/api.js`` holds the only ``fetch``)."""

from pathlib import Path

TEMPLATES = Path(__file__).resolve().parent / "templates"
STATIC = Path(__file__).resolve().parent / "static"
