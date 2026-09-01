"""Executable checks over the dataset, the split, the ranker and the served slates.

Every check states a claim, the value it expects, and the value actually measured
on this process. Nothing here is asserted in prose only: if a number appears in
the README or the UI, a check recomputes it.

Three verdicts are possible:

``pass``   the claim holds on this run
``fail``   the claim does not hold and something is wrong
``known``  the claim does not hold and that is a deliberate, documented trade-off

``known`` exists so the report does not have to choose between a dishonest green
board and a false alarm.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import registry
from .data import build_demo_dataset
from .recommender import FEATURE_NAMES, POLICIES, SignalWeave


ARTIFACT_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "evaluation.json"

# artifacts/evaluation.json records the canonical build. A run on any other seed is
# an experiment, and comparing it against that file says nothing useful.
CANONICAL_SEED = 42

GROUP_PURPOSE = {
    "dataset": "The generated events are reproducible and internally consistent.",
    "split": "The three time windows are disjoint, ordered, and free of item leakage.",
    "features": "The matrix handed to the ranker is finite, bounded, and carries signal.",
    "model": "The ranker is measured on data it did not fit, and its score is described honestly.",
    "retrieval": "Candidate generation obeys the contracts the slate depends on.",
    "evaluation": "Published metrics match a live recomputation.",
    "serving": "The model answering requests is the one that was registered, and it is accounted for.",
}


def _check(group: str, check_id: str, claim: str, expected: str, observed: str, status: str, detail: str = "") -> dict:
    return {
        "id": check_id,
        "group": group,
        "claim": claim,
        "expected": expected,
        "observed": observed,
        "status": status,
        "detail": detail,
    }


def _verdict(ok: bool) -> str:
    return "pass" if ok else "fail"


def _digest(events: list[dict]) -> str:
    payload = json.dumps(events, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _dataset_checks(engine: SignalWeave) -> list[dict]:
    events, items, users = engine.events, engine.items, engine.users

    replay_items, replay_users, replay_events = build_demo_dataset(engine.seed)
    live_digest = _digest(events)
    replay_digest = _digest(replay_events)

    item_ids = {item["item_id"] for item in items}
    user_ids = {user["user_id"] for user in users}
    orphans = sum(1 for e in events if e["user_id"] not in user_ids or e["item_id"] not in item_ids)

    pairs = [(e["user_id"], e["item_id"]) for e in events]
    duplicates = len(pairs) - len(set(pairs))

    timestamps = [e["occurred_at"] for e in events]
    unordered = sum(1 for a, b in zip(timestamps, timestamps[1:]) if a > b)

    actions = Counter(e["action"] for e in events)
    rarest_action, rarest_count = actions.most_common()[-1]
    rarest_share = rarest_count / len(events)

    per_user = Counter(e["user_id"] for e in events)
    counts = set(per_user.values())

    return [
        _check(
            "dataset", "determinism",
            "Rebuilding the dataset with the same seed reproduces it byte for byte.",
            f"sha256[:16] == {live_digest}",
            f"sha256[:16] == {replay_digest}",
            _verdict(live_digest == replay_digest and len(replay_items) == len(items) and len(replay_users) == len(users)),
            "Two independent generator runs in this process were hashed and compared.",
        ),
        _check(
            "dataset", "referential_integrity",
            "Every event points at a user and an item that exist in the catalog.",
            "0 orphan events", f"{orphans} orphan events", _verdict(orphans == 0),
        ),
        _check(
            "dataset", "no_repeat_exposure",
            "A learner never interacts with the same item twice.",
            "0 duplicate (user, item) pairs", f"{duplicates} duplicate pairs", _verdict(duplicates == 0),
            "The simulator draws without replacement. Real logs repeat, so repeat-exposure "
            "handling is untested by this dataset.",
        ),
        _check(
            "dataset", "global_time_order",
            "The event log is stored in chronological order.",
            "0 out-of-order adjacent pairs", f"{unordered} out-of-order pairs", _verdict(unordered == 0),
        ),
        _check(
            "dataset", "action_mix",
            "All four implicit actions occur often enough to be learnable.",
            "each action >= 5% of events",
            f"rarest is '{rarest_action}' at {rarest_share:.1%} ({dict(actions)})",
            _verdict(rarest_share >= 0.05),
        ),
        _check(
            "dataset", "uniform_activity",
            "Every learner contributes the same number of events.",
            "1 distinct per-user event count",
            f"{len(counts)} distinct counts {sorted(counts)}",
            "pass" if len(counts) == 1 else "known",
            "Uniform activity is a simulator artifact. Real traffic is heavy-tailed, so this "
            "dataset cannot exercise sparse-user behaviour.",
        ),
    ]


def _split_checks(engine: SignalWeave) -> list[dict]:
    history, ranker, evaluation = engine.history_events, engine.ranker_events, engine.evaluation_events
    total = len(engine.events)

    ids = [ {e["event_id"] for e in part} for part in (history, ranker, evaluation) ]
    overlap = len(ids[0] & ids[1]) + len(ids[1] & ids[2]) + len(ids[0] & ids[2])
    covered = len(ids[0] | ids[1] | ids[2])

    windows: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for name, part in (("history", history), ("ranker", ranker), ("evaluation", evaluation)):
        for event in part:
            windows[event["user_id"]][name].append(event["occurred_at"])

    time_violations = 0
    for user_windows in windows.values():
        for earlier, later in (("history", "ranker"), ("ranker", "evaluation")):
            if user_windows[earlier] and user_windows[later] and max(user_windows[earlier]) > min(user_windows[later]):
                time_violations += 1

    history_items: dict[str, set[str]] = defaultdict(set)
    for event in history:
        history_items[event["user_id"]].add(event["item_id"])
    item_leaks = sum(1 for event in evaluation if event["item_id"] in history_items[event["user_id"]])

    ratios = tuple(round(len(part) / total, 4) for part in (history, ranker, evaluation))
    ratio_ok = abs(ratios[0] - 0.60) <= 0.02 and abs(ratios[1] - 0.20) <= 0.02 and abs(ratios[2] - 0.20) <= 0.02

    cutoff = engine.history_cutoff.isoformat()
    later_than_cutoff = sum(1 for e in history if e["occurred_at"] > cutoff)

    return [
        _check(
            "split", "partition_is_exact",
            "The three windows partition the event log with no overlap and no loss.",
            f"0 shared events, {total} events covered",
            f"{overlap} shared events, {covered} events covered",
            _verdict(overlap == 0 and covered == total),
        ),
        _check(
            "split", "chronological_per_user",
            "For every learner, history precedes ranker fit, which precedes evaluation.",
            "0 users with a time inversion",
            f"{time_violations} users with a time inversion",
            _verdict(time_violations == 0),
            "This is the check that would catch training on the future.",
        ),
        _check(
            "split", "no_item_leakage",
            "No item scored in the frozen window was already interacted with in history.",
            "0 leaked items", f"{item_leaks} leaked items", _verdict(item_leaks == 0),
        ),
        _check(
            "split", "ratio_holds",
            "The split lands on 60 / 20 / 20 of each learner's stream.",
            "0.60 / 0.20 / 0.20 (+-2pp)",
            f"{ratios[0]} / {ratios[1]} / {ratios[2]}",
            _verdict(ratio_ok),
        ),
        _check(
            "split", "trending_uses_history_only",
            "The trending signal is built from history events only.",
            "0 history events after the cutoff",
            f"{later_than_cutoff} events after {cutoff}",
            _verdict(later_than_cutoff == 0),
        ),
    ]


def _feature_checks(engine: SignalWeave) -> list[dict]:
    matrix = engine.ranker_matrix
    labels = engine.ranker_labels
    non_finite = int((~np.isfinite(matrix)).sum())
    out_of_range = int(((matrix < -1e-9) | (matrix > 1 + 1e-9)).sum())
    positive_rate = float(labels.mean())

    spans = {name: float(matrix[:, i].max() - matrix[:, i].min()) for i, name in enumerate(FEATURE_NAMES)}
    narrow = {name: round(span, 3) for name, span in spans.items() if span < 0.30}

    coefficients = {row["feature"]: row["coefficient"] for row in engine.training_summary["coefficients"]}
    inert = {name: value for name, value in coefficients.items() if abs(value) < 0.03}

    return [
        _check(
            "features", "matrix_is_finite",
            "The ranker never sees NaN or infinity.",
            "0 non-finite cells", f"{non_finite} non-finite cells of {matrix.size}", _verdict(non_finite == 0),
        ),
        _check(
            "features", "features_bounded",
            "Every feature stays inside [0, 1] before standardisation.",
            "0 out-of-range cells", f"{out_of_range} out-of-range cells", _verdict(out_of_range == 0),
        ),
        _check(
            "features", "label_balance",
            "The fit set is not degenerate after negative sampling.",
            "positive rate in [0.15, 0.45]", f"{positive_rate:.4f} over {len(labels)} rows",
            _verdict(0.15 <= positive_rate <= 0.45),
        ),
        _check(
            "features", "features_vary",
            "Each feature actually varies across the fit set.",
            "every feature spans >= 0.30",
            "all features vary" if not narrow else f"narrow: {narrow}",
            "pass" if not narrow else "known",
            "A feature with almost no spread cannot separate classes no matter what "
            "coefficient it receives.",
        ),
        _check(
            "features", "features_carry_weight",
            "Every feature earns a non-trivial coefficient.",
            "every |coefficient| >= 0.03",
            "all features weighted" if not inert else f"inert: {inert}",
            "pass" if not inert else "known",
            "Inert features are kept because they are cheap and would matter on a catalog "
            "with real variance, but they are not doing work on this dataset.",
        ),
    ]


def _model_checks(engine: SignalWeave) -> list[dict]:
    summary = engine.training_summary
    auc = summary["cv_roc_auc"]
    gap = abs(summary["mean_held_out_score"] - summary["observed_positive_rate"])

    return [
        _check(
            "model", "beats_random_ordering",
            "The ranker orders held-out examples better than chance.",
            "5-fold ROC AUC > 0.55",
            f"{auc} (Brier {summary['cv_brier']}, {summary['rows']} rows)",
            _verdict(auc > 0.55),
            "Cross-validated inside the ranker-fit window; the frozen window is untouched.",
        ),
        _check(
            "model", "score_is_a_probability",
            "The sigmoid output can be read as an engagement probability.",
            "|mean score - observed rate| <= 0.05",
            f"{gap:.4f} (mean score {summary['mean_held_out_score']} vs rate {summary['observed_positive_rate']})",
            "known",
            summary["calibration_note"] + " Use it for ordering, not for expected-value maths.",
        ),
        _check(
            "model", "explanations_reconstruct_the_score",
            "Per-item feature contributions sum back to the logit that produced the score.",
            "max reconstruction error < 1e-9",
            *_explanation_fidelity(engine),
        ),
    ]


def _explanation_fidelity(engine: SignalWeave) -> tuple[str, str]:
    """Confirm the numbers shown in the UI are the numbers the model used."""

    worst = 0.0
    for user in engine.users[:8]:
        signals, sources = engine._retrieve(user["user_id"])
        ranked = engine._rank_candidates(signals, sources)
        for candidate in ranked.values():
            logit = float(np.log(candidate["relevance"] / (1.0 - candidate["relevance"])))
            rebuilt = float(engine.ranker.intercept_[0]) + float(candidate["contributions"].sum())
            worst = max(worst, abs(logit - rebuilt))
    return f"max error {worst:.2e}", _verdict(worst < 1e-9)


def _retrieval_checks(engine: SignalWeave) -> list[dict]:
    sample = [user["user_id"] for user in engine.users[:16]]

    signals = engine._signal_arrays(sample[0])
    exploration = 0.58 * (1.0 - signals["popularity"]) + 0.27 * signals["quality"] + 0.15 * signals["freshness"]
    vectors = {
        "content": signals["content"],
        "collaborative": signals["collaborative"],
        "trending": 0.7 * engine.trending + 0.3 * signals["popularity"],
        "exploration": exploration,
    }
    names = list(vectors)
    worst_pair, worst_corr = ("", ""), 0.0
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            corr = abs(float(np.corrcoef(vectors[left], vectors[right])[0, 1]))
            if corr > worst_corr:
                worst_pair, worst_corr = (left, right), corr

    seen_violations = 0
    cap_violations = 0
    short_slates = 0
    for user_id in sample:
        seen = engine.seen_by_user[user_id]
        for policy, config in POLICIES.items():
            rows = engine.recommend(user_id, policy=policy, limit=8)["recommendations"]
            if len(rows) != 8:
                short_slates += 1
            seen_violations += sum(1 for row in rows if row["item_id"] in seen)
            per_creator = Counter(row["creator"] for row in rows)
            cap_violations += sum(1 for count in per_creator.values() if count > config["creator_cap"])

    reachable, positives_total = 0, 0
    positives: dict[str, set[int]] = defaultdict(set)
    for event in engine.evaluation_events:
        if event["action"] in {"save", "complete"}:
            positives[event["user_id"]].add(engine.item_index[event["item_id"]])
    for user_id, truth in list(positives.items())[:32]:
        _, sources = engine._retrieve(user_id)
        reachable += len(truth & set(sources))
        positives_total += len(truth)
    candidate_recall = reachable / max(positives_total, 1)

    return [
        _check(
            "retrieval", "sources_are_not_redundant",
            "The four retrieval routes rank the catalog differently.",
            "max |correlation| between any two routes < 0.95",
            f"{worst_corr:.4f} between {worst_pair[0]} and {worst_pair[1]}",
            _verdict(worst_corr < 0.95),
            "Before trending was time-decayed this pair sat at 0.99, which made one route "
            "a duplicate of popularity.",
        ),
        _check(
            "retrieval", "seen_items_excluded",
            "Items already interacted with never reappear in a slate.",
            "0 repeats across the sample",
            f"{seen_violations} repeats over {len(sample)} learners x {len(POLICIES)} policies",
            _verdict(seen_violations == 0),
        ),
        _check(
            "retrieval", "creator_cap_enforced",
            "Each policy honours its own creator cap.",
            "0 slates over cap", f"{cap_violations} slates over cap", _verdict(cap_violations == 0),
        ),
        _check(
            "retrieval", "slate_size_honored",
            "A request for 8 items returns 8 items.",
            "0 short slates", f"{short_slates} short slates", _verdict(short_slates == 0),
        ),
        _check(
            "retrieval", "candidate_recall_ceiling",
            "Retrieval surfaces the frozen-window positives the ranker is asked to find.",
            "report the ceiling, target > 0.60",
            f"{candidate_recall:.4f} ({reachable}/{positives_total} positives reachable)",
            _verdict(candidate_recall > 0.60),
            "NDCG@10 cannot exceed this ceiling. Ranking work is wasted on anything retrieval "
            "never proposes.",
        ),
    ]


def _evaluation_checks(engine: SignalWeave) -> list[dict]:
    report = engine.report
    bounded = []
    for policy, metrics in report["policies"].items():
        for name, value in metrics.items():
            if not 0.0 <= value <= (10.0 if name == "novelty_bits" else 1.0):
                bounded.append(f"{policy}.{name}={value}")

    bootstrap = report["paired_bootstrap"]
    low, high = bootstrap["confidence_interval_95"]
    contains = low <= bootstrap["mean_delta"] <= high
    crosses_zero = low <= 0 <= high

    canonical = engine.seed == CANONICAL_SEED
    if not canonical:
        matches, observed = None, f"not applicable: this run is seed {engine.seed}, the artifact records seed {CANONICAL_SEED}"
    elif ARTIFACT_PATH.exists():
        stored = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
        matches = stored == report
        observed = "identical to this run" if matches else "differs from this run"
    else:
        matches, observed = False, "artifacts/evaluation.json is missing"

    gate = report["release_gate"]
    failed = [rule["id"] for rule in gate["guardrails"] if not rule["passed"]]

    unevaluated = sorted(set(POLICIES) - set(report["policies"]))
    baselines = {name: report["policies"][name]["ndcg_at_10"] for name in ("popularity", "content")}
    strongest = max(baselines, key=baselines.get)
    challenger = report["policies"]["balanced"]["ndcg_at_10"]
    margin = round(challenger - baselines[strongest], 4)

    return [
        _check(
            "evaluation", "metrics_in_range",
            "Every published metric is inside its mathematically valid range.",
            "0 impossible values", f"{len(bounded)} impossible values {bounded}", _verdict(not bounded),
        ),
        _check(
            "evaluation", "bootstrap_is_consistent",
            "The 95% interval brackets the mean it was resampled from.",
            "lower <= mean <= upper",
            f"[{low}, {high}] around {bootstrap['mean_delta']}", _verdict(contains),
        ),
        _check(
            "evaluation", "effect_excludes_zero",
            "The ranking gain over popularity is distinguishable from no gain.",
            "the 95% interval excludes 0",
            f"interval [{low}, {high}] {'includes' if crosses_zero else 'excludes'} 0",
            "known" if crosses_zero else "pass",
            "62 evaluable learners is a small sample; the interval is wide even when the point "
            "estimate is large.",
        ),
        _check(
            "evaluation", "artifact_matches_runtime",
            "The committed evaluation artifact is what this build actually produces.",
            "artifacts/evaluation.json == live report", observed,
            "known" if matches is None else _verdict(matches),
            "Regenerate with: python -m signalweave.train, which rewrites it as part of registering.",
        ),
        _check(
            "evaluation", "served_policies_are_evaluated",
            "Every policy the API can serve has frozen-window metrics of its own.",
            "0 served policies without metrics",
            f"{len(unevaluated)} without metrics {unevaluated}",
            _verdict(not unevaluated),
            "Otherwise the console would show one policy's slate next to another policy's numbers.",
        ),
        _check(
            "evaluation", "challenger_beats_strongest_baseline",
            "The promoted challenger outranks the best baseline, not just the weakest one.",
            f"balanced NDCG@10 >= {strongest} ({baselines[strongest]})",
            f"balanced {challenger}, margin {margin:+}",
            "pass" if margin >= 0 else "known",
            "The release gate compares against popularity only. Balanced trades ranking "
            "accuracy for slate diversity on purpose, so this gap is a product choice — but it "
            "is a gap, and the gate does not test for it.",
        ),
        _check(
            "evaluation", "release_gate_agrees",
            "The release status follows from the individual guardrails.",
            "status is promote_to_shadow only when every guardrail passes",
            f"status={gate['status']}, failing={failed or 'none'}",
            _verdict((gate["status"] == "promote_to_shadow") == (not failed)),
        ),
    ]


def _serving_checks(engine: SignalWeave) -> list[dict]:
    """Only meaningful once a registered model is being served."""

    bundle = engine.bundle
    manifest = bundle.manifest
    live_model = registry.model_digest(engine.ranker)
    live_data = registry.data_digest(engine.events)
    live_code = registry.code_digest()
    metrics_match = engine.report is bundle.report and engine.report["policies"] == manifest["metrics"]

    return [
        _check(
            "serving", "model_came_from_the_registry",
            "The engine is serving a registered version rather than an in-process fit.",
            "a version identifier", f"{bundle.version} (seed {manifest['seed']})", "pass",
            "Serving never fits on the request path; a training run is the only thing that "
            "produces coefficients.",
        ),
        _check(
            "serving", "binary_matches_its_manifest",
            "The loaded coefficients hash to the digest recorded when they were registered.",
            f"model_digest == {manifest['model_digest']}", f"{live_model}",
            _verdict(live_model == manifest["model_digest"]),
            "Catches a bundle that was swapped, truncated, or written by a different run.",
        ),
        _check(
            "serving", "metrics_belong_to_the_served_model",
            "The published policy metrics are the ones this model scored, not a recomputation.",
            "report is the bundle report",
            "loaded from the bundle" if metrics_match else "report does not match the manifest",
            _verdict(metrics_match),
        ),
        _check(
            "serving", "data_has_not_moved_under_the_model",
            "The generator still produces the dataset the champion was fitted on.",
            f"data_digest == {manifest['data_digest']}", f"{live_data}",
            _verdict(live_data == manifest["data_digest"]),
            "A change here means the model is scoring a population it was not fitted on.",
        ),
        _check(
            "serving", "serving_code_matches_training_code",
            "data.py and recommender.py are the revision that produced this model.",
            f"code_digest == {manifest['code_digest']}", f"{live_code}",
            "pass" if live_code == manifest["code_digest"] else "known",
            "Serving a model trained by older code is normal between deploys. What matters is "
            "that it is visible: run `python -m signalweave.train` to bring them back in line.",
        ),
    ]


def run_validation(engine: SignalWeave) -> dict:
    """Run every check against a live engine and return a serialisable report."""

    groups = [
        ("dataset", _dataset_checks(engine)),
        ("split", _split_checks(engine)),
        ("features", _feature_checks(engine)),
        ("model", _model_checks(engine)),
        ("retrieval", _retrieval_checks(engine)),
        ("evaluation", _evaluation_checks(engine)),
    ]
    if engine.bundle is not None:
        groups.append(("serving", _serving_checks(engine)))
    checks = [check for _, group_checks in groups for check in group_checks]
    tally = Counter(check["status"] for check in checks)
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": engine.seed,
        "summary": {
            "checks": len(checks),
            "passed": tally["pass"],
            "failed": tally["fail"],
            "known_limitations": tally["known"],
        },
        "groups": [
            {"name": name, "purpose": GROUP_PURPOSE[name], "checks": group_checks}
            for name, group_checks in groups
        ],
    }


def _print_report(report: dict) -> None:
    mark = {"pass": "PASS", "fail": "FAIL", "known": "KNOWN"}
    for group in report["groups"]:
        print(f"\n{group['name'].upper()}  —  {group['purpose']}")
        for check in group["checks"]:
            print(f"  [{mark[check['status']]:5s}] {check['id']}")
            print(f"          claim    {check['claim']}")
            print(f"          expected {check['expected']}")
            print(f"          observed {check['observed']}")
    summary = report["summary"]
    print(
        f"\n{summary['checks']} checks: {summary['passed']} pass, "
        f"{summary['failed']} fail, {summary['known_limitations']} known limitations"
    )


def main() -> int:
    import sys

    engine = SignalWeave()
    report = run_validation(engine)
    _print_report(report)

    if "--write" in sys.argv:
        ARTIFACT_PATH.parent.mkdir(exist_ok=True)
        ARTIFACT_PATH.write_text(json.dumps(engine.report, indent=2) + "\n", encoding="utf-8")
        (ARTIFACT_PATH.parent / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {ARTIFACT_PATH} and {ARTIFACT_PATH.parent / 'validation.json'}")

    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
