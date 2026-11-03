"""The home page's dashboard (v5.4 F10): two parts side by side, and the session.

1. scans: the newest scan of each target, the riskiest first, and the newest scans;
2. runs: the newest saved runs;
3. session: what this browser session holds now (a dataset, a training, a bundle, the last
   predictions, a scan); the predictions are kept in the session only.

The two tools stay apart: nothing links a scan to a model, and no IDS model is suggested from
open ports or CVEs (the review's objection to F7 as first planned; ROADMAP section 5).
See also: core/services/recon/saved_scans.py (the risks), core/services/ids/comparison.py
(the runs), ui/static/js/dashboard.js (the page).
"""

from __future__ import annotations

from typing import Any

from ...core.services.ids.comparison import saved_runs
from ...core.services.recon.saved_scans import RISK_ORDER, saved_scans, targets_at_risk
from .jobs import SCAN, TRAINING, Job
from .workspace import Workspace

RECENT = 5  # scans and runs listed as the newest
PREDICTION_KEYS = ("source", "model", "file", "rows", "counts")


def _unreadable(what: str, error: OSError) -> dict[str, str]:
    """A part that couldn't be read, said in a line."""
    return {"error": f"The saved {what} couldn't be read: {error.strerror or error}."}


def scans_part() -> dict[str, Any]:
    """The targets at risk among the newest saved scans (``saved_scans``' first page), how
    many of them have each risk, and the newest scans."""
    try:
        answer = saved_scans()
    except OSError as error:
        return _unreadable("scans", error)
    targets = targets_at_risk(answer["scans"])
    risks = [target["risk"] for target in targets]
    return {"targets": targets,
            "by_risk": [{"risk": r, "count": risks.count(r)} for r in RISK_ORDER if r in risks],
            "recent": answer["scans"][:RECENT], "read": len(answer["scans"]),
            "more": answer["more"], "left_out": answer["left_out"]}


def runs_part() -> dict[str, Any]:
    """The newest saved runs, with their selected model and its scores."""
    try:
        answer = saved_runs(limit=RECENT)
    except OSError as error:
        return _unreadable("runs", error)
    return {"recent": answer["runs"], "more": answer["more"], "left_out": answer["left_out"]}


def _job(job: Job | None) -> dict[str, Any] | None:
    """A job still queued or running: its state and where it is."""
    return {"state": job.state, "phase": job.phase()} if job is not None else None


def session_part(ws: Workspace) -> dict[str, Any]:
    """What the session holds now, each part None when it has none."""
    data, run, bundle = ws.data, ws.run, ws.loaded
    found = ws.predictions.answer if ws.predictions is not None else None
    last = ws.scan.answer if ws.scan is not None else None
    return {
        "dataset": {"files": list(data.files), "rows": len(data.frame)} if data else None,
        "training": _job(ws.active_job(TRAINING)),
        "run": {"run_id": run.run_id, "saved": ws.not_saved is None} if run else None,
        "bundle": {"run_id": bundle.run_id, "models": len(bundle.models)} if bundle else None,
        "predictions": {key: found.get(key) for key in PREDICTION_KEYS} if found else None,
        "scan": _job(ws.active_job(SCAN)),
        "last_scan": {"target": last.get("target"), "open_ports": len(last.get("entries", [])),
                      "scan_id": (last.get("saved") or {}).get("scan_id")} if last else None,
    }
