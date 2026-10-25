"""Write ``outputs.json``: what the data functions produce for the fixtures, made **once** by
the code before the v5.2 step 2.3 refactor (step 2.3a), so the refactored code is compared
with the old code's results, never with its own.

It refuses to replace an existing file: a golden written by the code under test would prove
nothing. ``--force`` only for a deliberate, reviewed change of behaviour.

Usage (from the project root):  python tests/fixtures/golden/make_goldens.py [--force]
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent.parent.parent), str(HERE.parent.parent)]

from golden_outputs import outputs  # noqa: E402

TARGET = HERE / "outputs.json"


def main() -> None:
    if TARGET.exists() and "--force" not in sys.argv[1:]:
        sys.exit(f"{TARGET.name} exists; it is written once, from the code before the change.")
    data = outputs(HERE.parent)
    TARGET.write_text(json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"{TARGET.name}: {len(data['files'])} files, {len(data['rule_table'])} rule rows")


if __name__ == "__main__":
    main()
