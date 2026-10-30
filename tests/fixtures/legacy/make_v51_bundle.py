"""Make ``v51_names.bundle``: a bundle saved by the v5.1 layout, before the code was arranged
by interface (v5.2 step 2.3), whose pickle refers to ``toolkit.services.ids.pipeline``
(``FittedModel``, ``_finite``, ``_signed_log``) and ``toolkit.services.ids.registry``
(``_TopK``, ``_Scores``). Every bundle saved from v5.1 step 1.5d to v5.2 step 2.3a has these
names. ``tests/ml/test_legacy_bundle.py`` proves it still loads.

It only makes that file with the v5.1 layout: run it from a checkout of commit
"test: tests that pin today's behaviour (step 2.3a)" or earlier (but after step 1.5d). On
newer code it stops: ``toolkit.services`` no longer exists, and the pickle would carry the
new names.

Usage (from the project root):  python tests/fixtures/legacy/make_v51_bundle.py
"""

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))

try:
    from toolkit.services.ids.bundle import _layout, pack
    from toolkit.services.ids.runs import DatasetRef, TrainSettings, train
except ModuleNotFoundError:
    sys.exit("toolkit.services.ids does not exist: run this from the v5.1 layout.")

KEY = bytes(range(32))  # the test key; the bundle is signed with it
CLASSES = ("BENIGN", "DoS", "PortScan")
V51_NAMES = (b"toolkit.services.ids.pipeline", b"toolkit.services.ids.registry")


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
                source=DatasetRef(name="synthetic.csv", sha256="cd" * 32, rows=400, columns=5))
    data = pack(run, KEY)
    pickled = zlib.decompress(_layout(data)[1])  # joblib's zlib stream; nothing unpickled
    missing = [name.decode() for name in V51_NAMES if name not in pickled]
    if missing:
        sys.exit(f"Not a v5.1-names bundle: the pickle has no {', '.join(missing)}. "
                 "Run this from the v5.1 layout.")
    sample = frame(30, 7).drop(columns="Label")
    expected = {name: [str(label) for label in fitted.predict(sample)]
                for name, fitted in run.evaluation.training.models.items()}
    (HERE / "v51_names.bundle").write_bytes(data)
    (HERE / "v51_names.expected.json").write_text(
        json.dumps({"run_id": run.run_id, "rows": sample.to_dict(orient="records"),
                    "predictions": expected}, indent=1) + "\n", encoding="utf-8")
    print(f"v51_names.bundle: {len(data):,} bytes, models {', '.join(expected)}")


if __name__ == "__main__":
    main()
