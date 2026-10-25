"""Make ``old_names.bundle``: a bundle saved by the code before the layered layout (v5.1
step 1.5d), whose pickle refers to the old module names ``toolkit.ids_core.pipeline``
(``FittedModel``, ``_finite``, ``_signed_log``) and ``toolkit.ids_core.registry``
(``_TopK``, ``_Scores``). ``tests/ml/test_legacy_bundle.py`` proves it still loads.

It only makes that file with the old code: run it from a checkout of commit
"test: an old bundle keeps loading (step 1.5d)" or earlier. On newer code it stops,
because the pickle would carry the new names.

Usage (from the project root):  python tests/fixtures/legacy/make_legacy_bundle.py
"""

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))

from toolkit.ids_core.bundle import _layout, pack  # noqa: E402
from toolkit.ids_core.runs import DatasetRef, TrainSettings, train  # noqa: E402

KEY = bytes(range(32))  # the test key; the bundle is signed with it
CLASSES = ("BENIGN", "DoS", "PortScan")
OLD_NAMES = (b"toolkit.ids_core.pipeline", b"toolkit.ids_core.registry")


def frame(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = rng.choice(CLASSES, size=rows, p=[0.6, 0.25, 0.15])
    shift = pd.Series(label).map({"BENIGN": 0.0, "DoS": 2.0, "PortScan": -2.0}).to_numpy()
    return pd.DataFrame({"a": shift + rng.normal(size=rows), "b": rng.normal(size=rows),
                         "c": rng.normal(size=rows) * 100,
                         "proto": rng.choice(["tcp", "udp"], size=rows), "Label": label})


def main() -> None:
    # logistic regression: signed log and scaling; f_classif: the _TopK selector
    settings = TrainSettings(label="Label", strategy="stratified",
                             models=("decision_tree", "logistic_regression"),
                             selector="f_classif", k_features=3, workers=1)
    run = train(frame(400, 0), settings=settings,
                source=DatasetRef(name="synthetic.csv", sha256="ab" * 32, rows=400, columns=5))
    data = pack(run, KEY)
    pickled = zlib.decompress(_layout(data)[1])  # joblib's zlib stream; nothing unpickled
    missing = [name.decode() for name in OLD_NAMES if name not in pickled]
    if missing:
        sys.exit(f"Not an old-names bundle: the pickle has no {', '.join(missing)}. "
                 "Run this from the code before the layered layout.")
    sample = frame(30, 7).drop(columns="Label")
    expected = {name: [str(label) for label in fitted.predict(sample)]
                for name, fitted in run.evaluation.training.models.items()}
    (HERE / "old_names.bundle").write_bytes(data)
    (HERE / "old_names.expected.json").write_text(
        json.dumps({"run_id": run.run_id, "rows": sample.to_dict(orient="records"),
                    "predictions": expected}, indent=1) + "\n", encoding="utf-8")
    print(f"old_names.bundle: {len(data):,} bytes, models {', '.join(expected)}")


if __name__ == "__main__":
    main()
