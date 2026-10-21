"""Dynamic IDS web trainer: starts ``toolkit.web`` (the app moved there in v5.1 step 1.2a).

This file stays as the launcher's entry point until the launcher moves (step 1.4).
Settings: ``TOOLKIT_PORT``, ``TOOLKIT_DEBUG``, ``TOOLKIT_MAX_UPLOAD_MB`` and the other
``TOOLKIT_*`` variables (see ``toolkit.config.web_config``).
"""

import sys
from pathlib import Path

# The toolkit package lives at the repo root and isn't installed yet (v5.1 packaging).
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolkit.web import run_server  # noqa: E402

if __name__ == "__main__":
    run_server()
