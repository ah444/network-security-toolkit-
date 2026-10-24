"""``python -m toolkit.web`` (or ``python -m toolkit web``): the web trainer on this
computer only (see ``run`` and ``run_server``).

Settings: ``TOOLKIT_PORT``, ``TOOLKIT_DEBUG``, ``TOOLKIT_MAX_UPLOAD_MB`` and the other
``TOOLKIT_*`` variables (``toolkit.data.settings.web_config``). Moved from
``tools/IDS/IDS_Web_CIC_Dynamic/app.py`` in v5.1 step 1.4a.
"""

from . import run

run()
