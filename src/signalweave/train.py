"""Training pipeline: fit, evaluate, validate, then register — in that order.

Run it as a command:

    python -m signalweave.train                     fit and register a new champion
    python -m signalweave.train --note "..."        with a reason attached
    python -m signalweave.train --list              show registered versions
    python -m signalweave.train --promote VERSION   repoint serving, i.e. roll back

A run that produces a failing check does not get registered. The gate between
"a model exists" and "a model is servable" is the check suite, not a person
remembering to look.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import registry
from .recommender import CHAMPION_POLICY, SignalWeave
from .validation import ARTIFACT_PATH, CANONICAL_SEED, run_validation


def serving_baseline(engine: SignalWeave, limit: int = 8) -> dict:
    """What a normal slate scores at training time, so drift has a reference."""

    scores, relevances = [], []
    for user in engine.users:
        rows = engine.recommend(user["user_id"], policy=CHAMPION_POLICY, limit=limit)["recommendations"]
        scores.extend(row["score"] for row in rows)
        relevances.extend(row["relevance"] for row in rows)
    return {
        "policy": CHAMPION_POLICY,
        "limit": limit,
        "users": len(engine.users),
        "mean_slate_score": round(float(np.mean(scores)), 4),
        "mean_ranker_score": round(float(np.mean(relevances)), 4),
    }


def train(seed: int = 42, note: str = "", promote: bool = True, allow_failing: bool = False) -> dict:
    """Fit a ranker, prove it, and register it."""

    print(f"fitting seed={seed} ...")
    engine = SignalWeave(seed=seed)

    print("running checks ...")
    validation = run_validation(engine)
    summary = validation["summary"]
    print(
        f"  {summary['checks']} checks: {summary['passed']} pass, "
        f"{summary['failed']} fail, {summary['known_limitations']} known"
    )
    if summary["failed"] and not allow_failing:
        failing = [
            check["id"]
            for group in validation["groups"]
            for check in group["checks"]
            if check["status"] == "fail"
        ]
        raise SystemExit(
            f"refusing to register: {len(failing)} failing check(s): {', '.join(failing)}\n"
            "Fix them, or pass --allow-failing-checks to register a known-bad model deliberately."
        )

    print("measuring serving baseline ...")
    baseline = serving_baseline(engine)

    manifest = registry.register(engine, validation=validation, note=note, promote=promote, baseline=baseline)

    if seed == CANONICAL_SEED:
        # artifacts/ is the committed record of the canonical build. A seed sweep is an
        # experiment: it earns a registry entry, not a rewrite of the repository's evidence.
        ARTIFACT_PATH.write_text(json.dumps(engine.report, indent=2) + "\n", encoding="utf-8")
        (ARTIFACT_PATH.parent / "validation.json").write_text(
            json.dumps(validation, indent=2) + "\n", encoding="utf-8"
        )
    else:
        print(f"  (seed {seed} is not the canonical build, so artifacts/ was left alone)")

    print(f"\nregistered {manifest['version']}{' (champion)' if promote else ''}")
    print(f"  data   {manifest['data_digest']}")
    print(f"  code   {manifest['code_digest']}")
    print(f"  model  {manifest['model_digest']}")
    print(f"  auc    {manifest['training']['cv_roc_auc']}  ndcg@10 {manifest['metrics']['balanced']['ndcg_at_10']}")
    print(f"  gate   {manifest['release_gate']}")
    return manifest


def load_or_bootstrap(seed: int = 42) -> SignalWeave:
    """The engine the API serves.

    Loads the champion when one exists. On a cold checkout there is nothing to
    load, so the first run trains and registers one — after that, startup is a
    load and the service never fits a model on the request path.
    """

    from .recommender import FEATURE_NAMES

    try:
        bundle = registry.load(feature_names=FEATURE_NAMES)
    except registry.IncompatibleBundle as error:
        print(f"registry: {error}\nregistry: retraining because the feature contract moved")
        bundle = None

    if bundle is not None:
        # Serve the data the model was fitted for, not whatever the caller defaulted to.
        return SignalWeave(seed=bundle.manifest.get("seed", seed), bundle=bundle)

    print("registry: no champion found, training an initial model ...")
    engine = SignalWeave(seed=seed)
    registry.register(
        engine,
        validation=run_validation(engine),
        note="bootstrap: registered automatically on first start",
        baseline=serving_baseline(engine),
    )
    return SignalWeave(seed=seed, bundle=registry.load(feature_names=FEATURE_NAMES))


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m signalweave.train", description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--note", default="", help="why this model was trained")
    parser.add_argument("--no-promote", action="store_true", help="register without making it champion")
    parser.add_argument("--allow-failing-checks", action="store_true")
    parser.add_argument("--list", action="store_true", help="list registered versions")
    parser.add_argument("--promote", metavar="VERSION", help="point serving at an existing version")
    args = parser.parse_args()

    if args.list:
        state = registry.read_registry()
        versions = registry.list_versions()
        if not versions:
            print("no registered models")
            return 0
        print(f"{'':2s}{'version':<26s}{'auc':>7s}{'ndcg@10':>9s}{'gate':>20s}  note")
        for manifest in versions:
            marker = "* " if manifest["version"] == state.get("champion") else "  "
            print(
                f"{marker}{manifest['version']:<26s}"
                f"{manifest['training']['cv_roc_auc']:>7.4f}"
                f"{manifest['metrics']['balanced']['ndcg_at_10']:>9.4f}"
                f"{manifest['release_gate']:>20s}  {manifest.get('note', '')}"
            )
        print("\n* = champion")
        return 0

    if args.promote:
        result = registry.promote(args.promote)
        print(f"champion {result['previous']} -> {result['champion']}")
        return 0

    train(
        seed=args.seed,
        note=args.note,
        promote=not args.no_promote,
        allow_failing=args.allow_failing_checks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
