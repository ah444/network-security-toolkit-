"""Signed model bundles (plan D11, step 0.4f; S1, S17, B18).

A bundle is one file holding the fitted models of a run, a JSON manifest that embeds the
run's TrainingRun record, and a signature:

    MAGIC | manifest length (8 bytes) | manifest JSON | payload length (8 bytes) |
    payload (joblib) | HMAC-SHA256 of everything before it (32 bytes)

Signing does not make unpickling safe; it only proves who made the file. So nothing is
deserialized until every check has passed: size and layout, then the signature (this
install's key) or a pinned SHA-256 (models shipped with the project), then the manifest,
then the library-version policy. Only then does ``joblib.load`` run.

This module works on bytes only; ``toolkit.bundle_files`` keeps the key and the files.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np

from .pipeline import FittedModel
from .registry import MODELS
from .runs import TrainingRun, package_versions

MAGIC = b"NSTOOLKIT-BUNDLE\n"
FORMAT = "nst-bundle"
FORMAT_VERSION = 1
KEY_BYTES = 32
_TAG_BYTES = hashlib.sha256().digest_size
_LENGTH_BYTES = 8
HEADER_BYTES = len(MAGIC) + _LENGTH_BYTES  # what precedes the manifest

# Version policy [B18], checked before deserializing:
#  estimator libraries the models use: major/minor differ → refuse, patch → warn
#  numpy / pandas: major differs → refuse, anything else → warn
#  python, scipy, joblib: major/minor differ → warn
ALWAYS_USED = ("scikit-learn", "imbalanced-learn")  # every pipeline is an imblearn Pipeline
ARRAY_LIBRARIES = ("numpy", "pandas")
WARN_LIBRARIES = ("python", "scipy", "joblib")


class BundleError(ValueError):
    """A bundle that is not loaded; ``code`` says why (e.g. ``BAD_SIGNATURE``)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class BundleVersionWarning(UserWarning):
    """The bundle was made with slightly different library versions."""


@dataclass(frozen=True, eq=False)
class Bundle:
    sha256: str  # of the whole file
    trust: str  # "signature" (this install's key) or "pinned" (SHA-256 pinned in the repo)
    manifest: dict[str, Any]
    models: dict[str, FittedModel]
    warnings: tuple[str, ...] = ()

    @property
    def run(self) -> dict[str, Any]:
        """The embedded TrainingRun record (its ``bundle_sha256`` is None: it is inside)."""
        return self.manifest["run"]

    @property
    def run_id(self) -> str:
        return self.manifest["run_id"]

    @property
    def selected(self) -> str:
        return self.manifest["selected"]

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(self.manifest["features"])

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(self.manifest["classes"])

    def model(self, name: str | None = None) -> FittedModel:
        return self.models[name or self.selected]


def installed_versions() -> dict[str, str | None]:
    return package_versions()


# --- pack -------------------------------------------------------------------------------

def pack(run: TrainingRun, key: bytes) -> bytes:
    """The run's fitted models as signed bundle bytes (the caller writes them)."""
    _check_key(key)
    models = dict(run.evaluation.training.models)
    if not models:
        raise ValueError("The run has no fitted models to bundle")
    buffer = io.BytesIO()
    joblib.dump(models, buffer, compress=3)
    payload = buffer.getvalue()
    selected = run.evaluation.selected if run.evaluation.selected in models else next(iter(models))
    libraries = set(ALWAYS_USED)
    libraries.update(MODELS[name].requires for name in models
                     if name in MODELS and MODELS[name].requires)
    manifest = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run.run_id,
        "selected": selected,
        "models": list(models),
        "features": list(run.evaluation.training.features),
        "classes": list(models[selected].classes),
        "libraries": sorted(libraries),
        "versions": installed_versions(),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_bytes": len(payload),
        "examples": examples(run),
        "run": run.to_dict(),
    }
    body = json.dumps(manifest, ensure_ascii=False, allow_nan=False).encode("utf-8")
    unsigned = b"".join([MAGIC, len(body).to_bytes(_LENGTH_BYTES, "big"), body,
                         len(payload).to_bytes(_LENGTH_BYTES, "big"), payload])
    return unsigned + hmac.new(key, unsigned, hashlib.sha256).digest()


def examples(run: TrainingRun) -> list[dict[str, Any]]:
    """One real row per class from the run's **test split** (complete, finite values), for
    trying a model out (the CLI's "test with a real example"). Never training rows."""
    label = run.settings.label or (run.profile.label_column if run.profile else None)
    test, features = run.split.test, list(run.evaluation.training.features)
    if not label or label not in test.columns:
        return []
    values = test[features]
    numeric = [c for c in features if values[c].dtype.kind in "iufb"]
    usable = values.notna().all(axis=1)
    if numeric:
        usable &= np.isfinite(values[numeric].to_numpy(dtype=float)).all(axis=1)
    labels = test[label].astype(str)
    found = []
    for cls in sorted(labels[usable].unique()):
        row = values[usable & (labels == cls)].iloc[0]
        found.append({"label": cls, "values": {c: (row[c].item() if hasattr(row[c], "item")
                                                    else str(row[c])) for c in features}})
    return found


# --- unpack -----------------------------------------------------------------------------

def unpack(data: bytes, *, key: bytes | None = None, pinned_sha256: str | None = None) -> Bundle:
    """Check ``data`` completely, then deserialize it. Trusted by ``pinned_sha256`` if given,
    else by the HMAC with ``key``. Raises BundleError before loading anything untrusted."""
    body, payload = _layout(data)
    sha256 = hashlib.sha256(data).hexdigest()
    if pinned_sha256 is not None:
        if not hmac.compare_digest(sha256, pinned_sha256.lower()):
            raise BundleError("HASH_MISMATCH", "The bundle's SHA-256 is not the pinned one: it "
                              "is not the file shipped with the project. Nothing was loaded.")
        trust = "pinned"
    elif key is not None:
        _check_key(key)
        expected = hmac.new(key, data[:-_TAG_BYTES], hashlib.sha256).digest()
        if not hmac.compare_digest(expected, data[-_TAG_BYTES:]):
            raise BundleError("BAD_SIGNATURE", "The bundle's signature does not match this "
                              "install's signing key: it was changed after it was saved, or it "
                              "was made on another install (bundles only load on the install "
                              "that made them). Nothing was loaded.")
        trust = "signature"
    else:
        raise BundleError("NO_TRUST", "A bundle is only loaded with this install's signing key "
                          "or a pinned SHA-256")

    manifest = _manifest(body)
    if hashlib.sha256(payload).hexdigest() != manifest.get("payload_sha256"):
        raise BundleError("BAD_PAYLOAD", "The bundle's model data does not match its manifest")
    notes = check_versions(manifest.get("versions") or {}, installed_versions(),
                           manifest.get("libraries") or ALWAYS_USED,
                           run_id=manifest.get("run_id"))
    for note in notes:
        warnings.warn(note, BundleVersionWarning, stacklevel=2)

    models = joblib.load(io.BytesIO(payload))  # trusted and version-checked from here on
    if (not isinstance(models, dict) or list(models) != list(manifest.get("models", ()))
            or not all(isinstance(m, FittedModel) for m in models.values())):
        raise BundleError("BAD_PAYLOAD", "The bundle's model data is not what its manifest lists")
    return Bundle(sha256=sha256, trust=trust, manifest=manifest, models=models,
                  warnings=tuple(notes))


def peek_manifest(data: bytes) -> dict[str, Any] | None:
    """The manifest **without** checking the signature, for messages only (never trust it)."""
    try:
        return _manifest(_layout(data)[0])
    except BundleError:
        return None


def manifest_length(head: bytes) -> int | None:
    """From a file's first HEADER_BYTES: how many manifest bytes follow (None: no bundle).
    With ``parse_manifest`` this lists bundles without reading their models."""
    if len(head) < HEADER_BYTES or not head.startswith(MAGIC):
        return None
    return int.from_bytes(head[len(MAGIC):HEADER_BYTES], "big")


def parse_manifest(body: bytes) -> dict[str, Any] | None:
    """The manifest JSON that follows the header, unchecked like ``peek_manifest``."""
    try:
        return _manifest(body)
    except BundleError:
        return None


def _layout(data: bytes) -> tuple[bytes, bytes]:
    if not data:
        raise BundleError("EMPTY", "The file is empty")
    if len(data) < len(MAGIC) or not data.startswith(MAGIC):
        raise BundleError("NOT_A_BUNDLE", "The file is not a toolkit bundle (models are only "
                          "loaded from bundles saved by this toolkit)")
    truncated = BundleError("TRUNCATED", "The bundle is incomplete (cut off while copying or "
                            "saving?)")
    position = len(MAGIC)
    lengths = []
    for _ in range(2):
        if len(data) < position + _LENGTH_BYTES:
            raise truncated
        length = int.from_bytes(data[position:position + _LENGTH_BYTES], "big")
        position += _LENGTH_BYTES
        lengths.append((position, length))
        position += length
    end = position + _TAG_BYTES
    if len(data) < end:
        raise truncated
    if len(data) > end:
        raise BundleError("EXTRA_DATA", "The bundle has extra bytes after its signature")
    (body_at, body_len), (payload_at, payload_len) = lengths
    return data[body_at:body_at + body_len], data[payload_at:payload_at + payload_len]


def _manifest(body: bytes) -> dict[str, Any]:
    try:
        manifest = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError("BAD_MANIFEST", "The bundle's manifest is not valid JSON") from error
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise BundleError("BAD_MANIFEST", "The bundle's manifest is not a toolkit manifest")
    version = manifest.get("format_version")
    if not isinstance(version, int) or version > FORMAT_VERSION:
        raise BundleError("NEWER_FORMAT", f"The bundle uses format {version}, newer than this "
                          f"version of the toolkit understands ({FORMAT_VERSION}); update it")
    return manifest


def _check_key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) < KEY_BYTES:
        raise BundleError("BAD_KEY", f"A signing key is {KEY_BYTES} random bytes")


# --- version policy [B18] -------------------------------------------------------------------

def _major_minor(version: str) -> tuple[int, int] | None:
    match = re.match(r"(\d+)(?:\.(\d+))?", version or "")
    return (int(match[1]), int(match[2] or 0)) if match else None


def check_versions(recorded: Mapping[str, str | None], installed: Mapping[str, str | None],
                   libraries: Sequence[str], *, run_id: str | None = None) -> list[str]:
    """Warnings for small differences; raises BundleError(VERSION_REFUSED) for risky ones."""
    retrain = "retrain from its TrainingRun" + (f" (run {run_id})" if run_id else "")
    notes = []
    for package in dict.fromkeys([*libraries, *ARRAY_LIBRARIES, *WARN_LIBRARIES]):
        made, have = recorded.get(package), installed.get(package)
        if made == have or made is None:
            continue
        if have is None:
            if package in libraries:
                raise BundleError("VERSION_REFUSED", f"The bundle needs {package} {made}, which "
                                  f"is not installed. Install it, or {retrain} without that "
                                  "model.")
            continue
        made_mm, have_mm = _major_minor(made), _major_minor(have)
        if package in libraries:
            refuse = made_mm is None or made_mm != have_mm
        elif package in ARRAY_LIBRARIES:
            refuse = made_mm is None or have_mm is None or made_mm[0] != have_mm[0]
        else:
            if made_mm is not None and made_mm == have_mm:
                continue
            refuse = False
        if refuse:
            series = f"{made_mm[0]}.{made_mm[1]}.x" if made_mm else made
            raise BundleError("VERSION_REFUSED", f"This bundle was made with {package} {made}, "
                              f"you have {have}: loading it could give wrong predictions, so it "
                              f"was not loaded. R{retrain[1:]}, or install {package} "
                              f"{series}.")
        notes.append(f"This bundle was made with {package} {made}, you have {have}; it loads, "
                     "but if predictions look wrong, retrain it.")
    return notes
