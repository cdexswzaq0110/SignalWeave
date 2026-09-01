"""A file-backed model registry.

The point of this module is one property: **serving does not train**. A training
run fits the ranker, evaluates it, validates it, and writes a versioned bundle.
The API loads that bundle and scores against it. The metrics on screen therefore
belong to the model actually answering requests, rather than to a re-derivation
that happens to run the same code.

Each version keeps two files:

``model.joblib``   the fitted objects and the fit matrix (binary, not in git)
``manifest.json``  what it is, what produced it, and how it scored (in git)

The manifest is the part worth keeping under version control: it records the data
digest, the digest of the source that defined the features, the environment, the
metrics and the check-suite outcome at training time. A binary you cannot account
for is not an asset.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
MODELS_ROOT = PROJECT_ROOT / "artifacts" / "models"
REGISTRY_PATH = MODELS_ROOT / "registry.json"

# Changing any of these files changes what a feature means, so a model trained
# before the change is not interchangeable with one trained after it.
CODE_FILES = ("data.py", "recommender.py")


class IncompatibleBundle(RuntimeError):
    """A stored bundle does not match the feature contract of the running code."""


@dataclass
class Bundle:
    """A registered model and everything needed to serve and audit it."""

    version: str
    manifest: dict
    scaler: object
    ranker: object
    feature_names: tuple[str, ...]
    fit_matrix: np.ndarray
    fit_labels: np.ndarray
    training_summary: dict
    report: dict

    @property
    def code_is_current(self) -> bool:
        return self.manifest.get("code_digest") == code_digest()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:16]


def code_digest() -> str:
    """Digest of the source that defines features, retrieval and the fit recipe."""

    joined = b"".join((PACKAGE_ROOT / name).read_bytes() for name in CODE_FILES)
    return _sha(joined)


def data_digest(events: list[dict]) -> str:
    return _sha(json.dumps(events, sort_keys=True).encode())


def model_digest(ranker) -> str:
    payload = np.concatenate([np.ravel(ranker.coef_), np.ravel(ranker.intercept_)])
    return _sha(np.round(payload, 10).tobytes())


def environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }


def read_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {"champion": None, "history": []}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _write_registry(state: dict) -> None:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def list_versions() -> list[dict]:
    """Every registered manifest, newest first."""

    if not MODELS_ROOT.exists():
        return []
    manifests = []
    for path in sorted(MODELS_ROOT.glob("*/manifest.json"), reverse=True):
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    return manifests


def register(engine, validation: dict | None = None, note: str = "", promote: bool = True,
             baseline: dict | None = None) -> dict:
    """Persist a trained engine as a new version and, by default, make it champion."""

    created = datetime.now(UTC)
    digest = model_digest(engine.ranker)
    version = f"{created.strftime('%Y%m%dT%H%M%SZ')}-{digest[:8]}"
    directory = MODELS_ROOT / version
    directory.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": version,
        "created_at": created.isoformat(timespec="seconds"),
        "seed": engine.seed,
        "data_digest": data_digest(engine.events),
        "code_digest": code_digest(),
        "model_digest": digest,
        "environment": environment(),
        "dataset": engine.report["dataset"],
        "training": engine.training_summary,
        "metrics": engine.report["policies"],
        "release_gate": engine.report["release_gate"]["status"],
        "validation": (validation or {}).get("summary"),
        "serving_baseline": baseline,
        "note": note,
    }

    joblib.dump(
        {
            "scaler": engine.scaler,
            "ranker": engine.ranker,
            "feature_names": tuple(engine.training_summary["features"]),
            "fit_matrix": engine.ranker_matrix,
            "fit_labels": engine.ranker_labels,
            "training_summary": engine.training_summary,
            "report": engine.report,
        },
        directory / "model.joblib",
        compress=3,
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    state = read_registry()
    state["history"] = [entry for entry in state.get("history", []) if entry["version"] != version]
    state["history"].insert(0, {"version": version, "registered_at": manifest["created_at"], "note": note})
    if promote:
        state["champion"] = version
    _write_registry(state)
    return manifest


def load(version: str | None = None, feature_names: tuple[str, ...] | None = None) -> Bundle | None:
    """Load a version, or the champion when none is named. Returns None if absent."""

    state = read_registry()
    target = version or state.get("champion")
    if not target:
        return None
    directory = MODELS_ROOT / target
    if not (directory / "model.joblib").exists():
        return None

    payload = joblib.load(directory / "model.joblib")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    stored_features = tuple(payload["feature_names"])

    if feature_names is not None and stored_features != tuple(feature_names):
        raise IncompatibleBundle(
            f"{target} was trained on {stored_features}, but this build expects {tuple(feature_names)}"
        )
    if payload["scaler"].n_features_in_ != len(stored_features):
        raise IncompatibleBundle(f"{target} scaler expects {payload['scaler'].n_features_in_} features")

    return Bundle(
        version=target,
        manifest=manifest,
        scaler=payload["scaler"],
        ranker=payload["ranker"],
        feature_names=stored_features,
        fit_matrix=payload["fit_matrix"],
        fit_labels=payload["fit_labels"],
        training_summary=payload["training_summary"],
        report=payload["report"],
    )


def promote(version: str) -> dict:
    """Point serving at an existing version. Rollback is a promote to an older one."""

    if not (MODELS_ROOT / version / "manifest.json").exists():
        raise KeyError(f"Unknown model version: {version}")
    state = read_registry()
    previous = state.get("champion")
    state["champion"] = version
    _write_registry(state)
    return {"champion": version, "previous": previous}


def registry_state(active: Bundle | None) -> dict:
    """What the operations view needs to describe the running model."""

    state = read_registry()
    versions = list_versions()
    if active is None:
        return {
            "champion": state.get("champion"),
            "loaded": None,
            "provenance": "unregistered",
            "code_is_current": True,
            "versions": versions,
            "note": "Serving an in-process fit. Run `python -m signalweave.train` to register one.",
        }
    return {
        "champion": state.get("champion"),
        "loaded": active.version,
        "provenance": "registry",
        "code_is_current": active.code_is_current,
        "code_digest_now": code_digest(),
        "manifest": active.manifest,
        "versions": versions,
        "note": (
            ""
            if active.code_is_current
            else "This model was trained by a different revision of data.py/recommender.py. "
            "Serving continues; retrain to bring them back in line."
        ),
    }
