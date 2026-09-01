"""Serving telemetry, shadow comparison, and drift on the live feedback log.

Three questions this answers, none of which the offline evaluation can:

1. Is the service fast enough, and is anything erroring?
2. If the challenger were promoted, how different would the slates be?
3. Has the traffic moved away from what the registered model was fitted on?

All of it runs on what the process already has: an in-memory ring buffer for
latency and the SQLite log for everything durable. Nothing here needs a metrics
backend, and nothing here pretends to be one.
"""

from __future__ import annotations

from collections import Counter, deque
from datetime import UTC, datetime

import numpy as np


# Below this many live events the drift numbers are noise, and saying so is more
# useful than printing a confident figure derived from nine data points.
MIN_EVENTS_FOR_DRIFT = 30

# New feedback events since the champion was fitted, after which a retrain is worth
# running. Arbitrary at this scale, and labelled as such wherever it is shown.
RETRAIN_AFTER_EVENTS = 50

ACTIONS = ("complete", "save", "open", "dismiss")


class ServingMetrics:
    """Fixed-size latency and error window, per route."""

    def __init__(self, capacity: int = 1000):
        self.samples: deque[tuple[str, float, int]] = deque(maxlen=capacity)
        self.started_at = datetime.now(UTC)
        self.total = 0

    def record(self, route: str, milliseconds: float, status: int) -> None:
        self.samples.append((route, milliseconds, status))
        self.total += 1

    def snapshot(self) -> dict:
        by_route: dict[str, list[tuple[float, int]]] = {}
        for route, milliseconds, status in self.samples:
            by_route.setdefault(route, []).append((milliseconds, status))

        routes = []
        for route, entries in sorted(by_route.items()):
            timings = np.asarray([value for value, _ in entries])
            routes.append(
                {
                    "route": route,
                    "requests": len(entries),
                    "p50_ms": round(float(np.percentile(timings, 50)), 1),
                    "p95_ms": round(float(np.percentile(timings, 95)), 1),
                    "max_ms": round(float(timings.max()), 1),
                    "errors": sum(1 for _, status in entries if status >= 400),
                }
            )
        return {
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "requests_total": self.total,
            "window": len(self.samples),
            "window_capacity": self.samples.maxlen,
            "routes": routes,
        }


def compare_slates(champion: list[dict], shadow: list[dict]) -> dict:
    """How far apart two policies are for one request."""

    champion_ids = [row["item_id"] for row in champion]
    shadow_ids = [row["item_id"] for row in shadow]
    k = max(len(champion_ids), 1)
    shared = set(champion_ids) & set(shadow_ids)

    shadow_rank = {item_id: index for index, item_id in enumerate(shadow_ids)}
    shifts = [abs(index - shadow_rank[item_id]) for index, item_id in enumerate(champion_ids) if item_id in shadow_rank]

    return {
        "k": len(champion_ids),
        "overlap": round(len(shared) / k, 4),
        "top1_agree": int(bool(champion_ids and shadow_ids and champion_ids[0] == shadow_ids[0])),
        "mean_rank_shift": round(float(np.mean(shifts)) if shifts else 0.0, 4),
        "champion_mean_score": round(float(np.mean([row["score"] for row in champion])) if champion else 0.0, 4),
    }


def population_stability_index(expected: dict[str, float], observed: dict[str, float]) -> float:
    """Standard PSI over a shared set of categories, with smoothing for empty bins."""

    epsilon = 1e-6
    total = 0.0
    for key in sorted(set(expected) | set(observed)):
        a = max(expected.get(key, 0.0), epsilon)
        b = max(observed.get(key, 0.0), epsilon)
        total += (b - a) * np.log(b / a)
    return round(float(total), 4)


def _share(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    return {key: value / total for key, value in counter.items()} if total else {}


def drift_report(engine, feedback_rows: list[tuple[str, str, str]], baseline: dict | None = None,
                 shadow_rows: list[dict] | None = None) -> dict:
    """Compare live feedback and live scores against what the model was built on."""

    trained_on = Counter(
        event["action"] for event in (engine.history_events + engine.ranker_events)
    )
    live = Counter(action for _, _, action in feedback_rows)
    live_total = sum(live.values())

    expected_share = _share(trained_on)
    observed_share = _share(live)
    psi = population_stability_index(expected_share, observed_share) if live_total else None

    if live_total < MIN_EVENTS_FOR_DRIFT:
        action_status = "insufficient_data"
        action_note = (
            f"{live_total} live events against a {MIN_EVENTS_FOR_DRIFT}-event minimum. "
            "A PSI computed on this little traffic would be noise."
        )
    elif psi is not None and psi >= 0.25:
        action_status = "drifted"
        action_note = "PSI at or above 0.25 is the conventional line for a materially different population."
    elif psi is not None and psi >= 0.10:
        action_status = "watch"
        action_note = "PSI between 0.10 and 0.25: worth watching, not yet worth retraining on."
    else:
        action_status = "stable"
        action_note = "Live action mix is close to the mix the ranker was fitted on."

    served_scores = [row["champion_mean_score"] for row in (shadow_rows or []) if row.get("champion_mean_score")]
    score_block = {"status": "insufficient_data", "observed": None, "baseline": None, "delta": None}
    if baseline and served_scores:
        observed = float(np.mean(served_scores))
        reference = float(baseline.get("mean_slate_score", 0.0))
        delta = observed - reference
        score_block = {
            "status": "stable" if abs(delta) <= 0.05 else "drifted",
            "observed": round(observed, 4),
            "baseline": round(reference, 4),
            "delta": round(delta, 4),
            "requests": len(served_scores),
        }

    return {
        "action_mix": {
            "status": action_status,
            "psi": psi,
            "note": action_note,
            "trained_on": {action: round(expected_share.get(action, 0.0), 4) for action in ACTIONS},
            "live": {action: round(observed_share.get(action, 0.0), 4) for action in ACTIONS},
            "live_events": live_total,
            "minimum_events": MIN_EVENTS_FOR_DRIFT,
        },
        "slate_score": score_block,
        "retrain": {
            "events_since_registration": live_total,
            "threshold": RETRAIN_AFTER_EVENTS,
            "recommended": live_total >= RETRAIN_AFTER_EVENTS,
            "note": (
                "Live feedback updates the learner profile immediately but never the ranker "
                "coefficients. Only a training run does that, and it produces a new version."
            ),
        },
    }


def shadow_summary(rows: list[dict], champion: str, shadow: str) -> dict:
    """Accumulated divergence between the served policy and the one in shadow."""

    if not rows:
        return {
            "champion_policy": champion,
            "shadow_policy": shadow,
            "requests": 0,
            "note": "No shadow comparisons recorded yet. Request a slate under the champion policy.",
        }

    overlap = np.asarray([row["overlap"] for row in rows])
    shift = np.asarray([row["mean_rank_shift"] for row in rows])
    champion_ms = np.asarray([row["champion_ms"] for row in rows])
    shadow_ms = np.asarray([row["shadow_ms"] for row in rows])

    return {
        "champion_policy": champion,
        "shadow_policy": shadow,
        "requests": len(rows),
        "mean_overlap": round(float(overlap.mean()), 4),
        "min_overlap": round(float(overlap.min()), 4),
        "top1_agreement": round(float(np.mean([row["top1_agree"] for row in rows])), 4),
        "mean_rank_shift": round(float(shift.mean()), 4),
        "champion_p95_ms": round(float(np.percentile(champion_ms, 95)), 1),
        "shadow_p95_ms": round(float(np.percentile(shadow_ms, 95)), 1),
        "note": (
            "The shadow policy is scored on every champion request and never shown. Overlap is the "
            "share of the champion slate it would keep; rank shift is how far the shared items move."
        ),
    }
