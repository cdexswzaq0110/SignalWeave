"""Hybrid retrieval, learning-to-rank, constrained slate optimization, and evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from math import log2

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import ACTION_WEIGHT, build_demo_dataset, temporal_split


FEATURE_NAMES = (
    "content affinity",
    "collaborative affinity",
    "topic affinity",
    "popularity",
    "freshness",
    "editorial quality",
    "difficulty fit",
    "duration fit",
)

# Trending decays engagement by event age. 21 days is roughly a quarter of the
# history window, so a burst two months old cannot outrank a current one.
TRENDING_HALF_LIFE_DAYS = 21.0

POLICIES = {
    "accuracy": {"relevance": 0.82, "diversity": 0.06, "novelty": 0.05, "freshness": 0.07, "creator_cap": 4},
    "balanced": {"relevance": 0.64, "diversity": 0.16, "novelty": 0.11, "freshness": 0.09, "creator_cap": 2},
    "discovery": {"relevance": 0.48, "diversity": 0.25, "novelty": 0.17, "freshness": 0.10, "creator_cap": 1},
}

# What production serves, and what runs behind it without being shown. Accuracy
# leads every ranking metric on the frozen window, so it is champion. The release
# gate returns promote_to_shadow for balanced, and this is where that verdict is
# acted on rather than just displayed.
CHAMPION_POLICY = "accuracy"
SHADOW_POLICY = "balanced"


def _minmax(values: np.ndarray) -> np.ndarray:
    low = float(values.min())
    span = float(values.max() - low)
    return np.zeros_like(values, dtype=float) if span < 1e-12 else (values - low) / span


class SignalWeave:
    """Small, inspectable recommendation engine with production-shaped boundaries."""

    def __init__(self, seed: int = 42, bundle=None):
        """Build the engine.

        With no bundle the ranker is fitted and evaluated in process, which is what
        a training run wants. Given a registered bundle the fitted objects and their
        frozen-window metrics are loaded instead: serving never re-fits, and the
        numbers it reports are the ones the served model actually scored.
        """

        self.seed = seed
        self.items, self.users, self.events = build_demo_dataset(seed)
        self.history_events, self.ranker_events, self.evaluation_events = temporal_split(self.events)
        self.item_index = {item["item_id"]: index for index, item in enumerate(self.items)}
        self.user_index = {user["user_id"]: index for index, user in enumerate(self.users)}
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.item_vectors = self.vectorizer.fit_transform(item["text"] for item in self.items)
        self.content_similarity = cosine_similarity(self.item_vectors)
        self.live_feedback: list[dict] = []
        self.bundle = bundle
        self._build_retrieval_state()
        if bundle is None:
            self._fit_ranker()
            self.report = self.evaluate()
        else:
            self._adopt_bundle(bundle)

    def _adopt_bundle(self, bundle) -> None:
        """Serve a previously registered model instead of fitting a new one."""

        if tuple(bundle.feature_names) != FEATURE_NAMES:
            raise ValueError(
                f"bundle feature contract {tuple(bundle.feature_names)} does not match {FEATURE_NAMES}"
            )
        self.scaler = bundle.scaler
        self.ranker = bundle.ranker
        self.ranker_matrix = bundle.fit_matrix
        self.ranker_labels = bundle.fit_labels
        self.training_summary = bundle.training_summary
        self.report = bundle.report

    @property
    def model_version(self) -> str:
        return self.bundle.version if self.bundle is not None else "unregistered"

    def _build_retrieval_state(self) -> None:
        user_item = np.zeros((len(self.users), len(self.items)), dtype=float)
        self.seen_by_user: dict[str, set[str]] = defaultdict(set)
        self.topic_affinity: dict[str, Counter] = defaultdict(Counter)
        popularity = np.zeros(len(self.items), dtype=float)
        recency = np.zeros(len(self.items), dtype=float)

        timestamps = {event["event_id"]: datetime.fromisoformat(event["occurred_at"]) for event in self.history_events}
        self.history_cutoff = max(timestamps.values(), default=datetime.now(UTC))
        for event in self.history_events:
            user_pos = self.user_index[event["user_id"]]
            item_pos = self.item_index[event["item_id"]]
            weight = max(0.0, float(event["weight"]))
            self.seen_by_user[event["user_id"]].add(event["item_id"])
            user_item[user_pos, item_pos] += weight
            popularity[item_pos] += weight
            if weight:
                topic = self.items[item_pos]["topic"]
                self.topic_affinity[event["user_id"]][topic] += weight
                event_age_days = (self.history_cutoff - timestamps[event["event_id"]]).total_seconds() / 86400.0
                recency[item_pos] += weight * 0.5 ** (event_age_days / TRENDING_HALF_LIFE_DAYS)

        row_norm = np.linalg.norm(user_item, axis=1, keepdims=True)
        normalized_user_item = user_item / np.maximum(row_norm, 1e-12)
        self.item_similarity = normalized_user_item.T @ normalized_user_item
        np.fill_diagonal(self.item_similarity, 0.0)
        self.user_item = user_item
        self.popularity = _minmax(np.log1p(popularity))
        self.trending = _minmax(np.log1p(recency))
        total_popularity = popularity.sum() + len(self.items)
        self.item_probability = (popularity + 1.0) / total_popularity

        self.user_profiles = np.zeros((len(self.users), self.item_vectors.shape[1]), dtype=float)
        for user_id, user_pos in self.user_index.items():
            weights = user_item[user_pos]
            if weights.sum() > 0:
                profile = self.item_vectors.multiply(weights[:, None]).sum(axis=0)
                dense = np.asarray(profile).ravel()
                self.user_profiles[user_pos] = dense / max(np.linalg.norm(dense), 1e-12)

    def _signal_arrays(self, user_id: str) -> dict[str, np.ndarray]:
        user = self.users[self.user_index[user_id]]
        user_pos = self.user_index[user_id]
        profile = self.user_profiles[user_pos].copy()
        live_seen: set[str] = set()
        for event in self.live_feedback:
            if event["user_id"] != user_id:
                continue
            item_pos = self.item_index[event["item_id"]]
            live_seen.add(event["item_id"])
            profile += max(0.0, ACTION_WEIGHT[event["action"]]) * self.item_vectors[item_pos].toarray().ravel()
        if np.linalg.norm(profile) > 0:
            profile /= np.linalg.norm(profile)

        content = np.asarray(self.item_vectors @ profile).ravel()
        collaborative = np.asarray(self.user_item[user_pos] @ self.item_similarity).ravel()
        collaborative = _minmax(collaborative)
        topic_counter = self.topic_affinity[user_id]
        topic_total = max(sum(topic_counter.values()), 1.0)
        topic = np.asarray([topic_counter[item["topic"]] / topic_total for item in self.items])
        freshness = np.asarray([max(0.0, 1.0 - item["age_days"] / 180) for item in self.items])
        quality = np.asarray([item["quality"] for item in self.items])
        difficulty = np.asarray([1.0 - abs(item["difficulty"] - user["level"]) / 2 for item in self.items])
        duration = np.asarray([max(0.0, 1.0 - abs(item["duration_min"] - user["time_budget_min"]) / 100) for item in self.items])
        return {
            "content": _minmax(content),
            "collaborative": collaborative,
            "topic": _minmax(topic),
            "popularity": self.popularity,
            "freshness": freshness,
            "quality": quality,
            "difficulty": difficulty,
            "duration": duration,
            "seen": self.seen_by_user[user_id] | live_seen,
        }

    def _feature_vector(self, signals: dict[str, np.ndarray], item_pos: int) -> np.ndarray:
        return np.asarray(
            [
                signals["content"][item_pos],
                signals["collaborative"][item_pos],
                signals["topic"][item_pos],
                signals["popularity"][item_pos],
                signals["freshness"][item_pos],
                signals["quality"][item_pos],
                signals["difficulty"][item_pos],
                signals["duration"][item_pos],
            ],
            dtype=float,
        )

    def _fit_ranker(self) -> None:
        rng = np.random.default_rng(self.seed)
        rows: list[np.ndarray] = []
        labels: list[int] = []
        for event in self.ranker_events:
            signals = self._signal_arrays(event["user_id"])
            item_pos = self.item_index[event["item_id"]]
            label = int(event["action"] in {"save", "complete"})
            rows.append(self._feature_vector(signals, item_pos))
            labels.append(label)
            if label:
                unseen = [index for index, item in enumerate(self.items) if item["item_id"] not in signals["seen"] and index != item_pos]
                for negative_pos in rng.choice(unseen, size=2, replace=False):
                    rows.append(self._feature_vector(signals, int(negative_pos)))
                    labels.append(0)

        matrix = np.vstack(rows)
        targets = np.asarray(labels)
        self.ranker_matrix = matrix
        self.ranker_labels = targets
        self.scaler = StandardScaler().fit(matrix)
        self.ranker = LogisticRegression(class_weight="balanced", max_iter=500, random_state=self.seed)
        self.ranker.fit(self.scaler.transform(matrix), targets)

        # The same recipe scored on held-out folds. Without this the ranker has no
        # measured quality at all: fitted coefficients on their own prove nothing.
        held_out = cross_val_predict(
            make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=500, random_state=self.seed),
            ),
            matrix,
            targets,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed),
            method="predict_proba",
        )[:, 1]
        self.training_summary = {
            "rows": len(rows),
            "positive_rate": round(float(targets.mean()), 4),
            "features": list(FEATURE_NAMES),
            "split": "60% representation history / 20% ranker fit / 20% frozen evaluation per user",
            "cv_folds": 5,
            "cv_roc_auc": round(float(roc_auc_score(targets, held_out)), 4),
            "cv_brier": round(float(brier_score_loss(targets, held_out)), 4),
            "mean_held_out_score": round(float(held_out.mean()), 4),
            "observed_positive_rate": round(float(targets.mean()), 4),
            "score_is_calibrated": False,
            "calibration_note": (
                "class_weight='balanced' reweights the classes, so the sigmoid output orders "
                "candidates but overstates absolute engagement probability."
            ),
            "coefficients": [
                {"feature": name, "coefficient": round(float(value), 4)}
                for name, value in sorted(
                    zip(FEATURE_NAMES, self.ranker.coef_[0]), key=lambda pair: -abs(pair[1])
                )
            ],
        }

    def _retrieve(self, user_id: str, per_source: int = 22) -> tuple[dict[str, np.ndarray], dict[int, set[str]]]:
        signals = self._signal_arrays(user_id)
        sources: dict[int, set[str]] = defaultdict(set)
        exploration = 0.58 * (1.0 - signals["popularity"]) + 0.27 * signals["quality"] + 0.15 * signals["freshness"]
        source_scores = {
            "content": signals["content"],
            "collaborative": signals["collaborative"],
            "trending": 0.7 * self.trending + 0.3 * signals["popularity"],
            "exploration": exploration,
        }
        for source, scores in source_scores.items():
            ordered = np.argsort(scores)[::-1]
            accepted = 0
            for item_pos in ordered:
                if self.items[int(item_pos)]["item_id"] in signals["seen"]:
                    continue
                sources[int(item_pos)].add(source)
                accepted += 1
                if accepted == per_source:
                    break
        return signals, sources

    def _rank_candidates(self, signals: dict[str, np.ndarray], sources: dict[int, set[str]]) -> dict[int, dict]:
        positions = sorted(sources)
        if not positions:
            return {}
        features = np.vstack([self._feature_vector(signals, item_pos) for item_pos in positions])
        scaled = self.scaler.transform(features)
        logits = float(self.ranker.intercept_[0]) + scaled @ self.ranker.coef_[0]
        contributions = scaled * self.ranker.coef_[0]
        scores = 1.0 / (1.0 + np.exp(-logits))
        return {
            item_pos: {
                "relevance": float(scores[row]),
                "features": features[row],
                "contributions": contributions[row],
                "sources": sorted(sources[item_pos]),
            }
            for row, item_pos in enumerate(positions)
        }

    def recommend(self, user_id: str, policy: str = "balanced", limit: int = 8) -> dict:
        if user_id not in self.user_index:
            raise KeyError(f"Unknown user: {user_id}")
        if policy not in POLICIES:
            raise KeyError(f"Unknown policy: {policy}")
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")

        signals, sources = self._retrieve(user_id)
        candidates = self._rank_candidates(signals, sources)
        weights = POLICIES[policy]
        selected: list[int] = []
        creator_counts: Counter = Counter()
        decision_rows: list[dict] = []

        while len(selected) < limit:
            best_pos = None
            best_utility = -1.0
            best_terms: dict[str, float] = {}
            runner_up_pos = None
            runner_up_utility = -1.0
            blocked_by_cap = 0
            for item_pos, candidate in candidates.items():
                if item_pos in selected:
                    continue
                creator = self.items[item_pos]["creator"]
                if creator_counts[creator] >= weights["creator_cap"]:
                    blocked_by_cap += 1
                    continue
                if selected:
                    diversity = 1.0 - float(self.content_similarity[item_pos, selected].max())
                else:
                    diversity = 1.0
                terms = {
                    "relevance": float(candidate["relevance"]),
                    "diversity": diversity,
                    "novelty": float(1.0 - signals["popularity"][item_pos]),
                    "freshness": float(signals["freshness"][item_pos]),
                }
                utility = sum(weights[term] * value for term, value in terms.items())
                if utility > best_utility:
                    runner_up_pos, runner_up_utility = best_pos, best_utility
                    best_pos, best_utility, best_terms = item_pos, utility, terms
                elif utility > runner_up_utility:
                    runner_up_pos, runner_up_utility = item_pos, utility
            if best_pos is None:
                break

            selected.append(best_pos)
            item = self.items[best_pos]
            creator_counts[item["creator"]] += 1
            candidate = candidates[best_pos]
            feature_contributions = sorted(
                zip(FEATURE_NAMES, candidate["contributions"]), key=lambda pair: pair[1], reverse=True
            )
            positive_reasons = [name for name, value in feature_contributions if value > 0][:2]
            if not positive_reasons:
                positive_reasons = ["balanced slate utility"]
            utility_terms = [
                {
                    "term": term,
                    "weight": weights[term],
                    "value": round(best_terms[term], 4),
                    "contribution": round(weights[term] * best_terms[term], 4),
                    "share": round(weights[term] * best_terms[term] / max(best_utility, 1e-12), 4),
                }
                for term in ("relevance", "diversity", "novelty", "freshness")
            ]
            decision_rows.append(
                {
                    **{key: value for key, value in item.items() if key != "text"},
                    "rank": len(selected),
                    "score": round(best_utility, 4),
                    "relevance": round(best_terms["relevance"], 4),
                    "diversity": round(best_terms["diversity"], 4),
                    "novelty": round(best_terms["novelty"], 4),
                    "freshness": round(best_terms["freshness"], 4),
                    "sources": candidate["sources"],
                    "why": f"Strong {positive_reasons[0]}" + (f" and {positive_reasons[1]}" if len(positive_reasons) > 1 else ""),
                    "utility_terms": utility_terms,
                    "decided_by": max(utility_terms, key=lambda row: row["contribution"])["term"],
                    "runner_up": None
                    if runner_up_pos is None
                    else {
                        "item_id": self.items[runner_up_pos]["item_id"],
                        "title": self.items[runner_up_pos]["title"],
                        "margin": round(best_utility - runner_up_utility, 4),
                    },
                    "blocked_by_creator_cap": blocked_by_cap,
                    "contributions": [
                        {
                            "feature": name,
                            "value": round(float(value), 4),
                            "raw": round(float(candidate["features"][FEATURE_NAMES.index(name)]), 4),
                        }
                        for name, value in feature_contributions
                    ],
                }
            )

        return {
            "user": self.users[self.user_index[user_id]],
            "policy": policy,
            "policy_config": weights,
            "catalog_size": len(self.items),
            "already_seen": len(signals["seen"]),
            "candidate_count": len(candidates),
            "retrieval_sources": {source: sum(source in row["sources"] for row in candidates.values()) for source in ("content", "collaborative", "trending", "exploration")},
            "recommendations": decision_rows,
        }

    def record_feedback(self, user_id: str, item_id: str, action: str) -> None:
        if user_id not in self.user_index or item_id not in self.item_index:
            raise KeyError("Unknown user or item")
        if action not in ACTION_WEIGHT:
            raise ValueError(f"Unsupported action: {action}")
        self.live_feedback.append({"user_id": user_id, "item_id": item_id, "action": action})

    def _slate_metrics(self, slates: dict[str, list[int]], positives: dict[str, set[int]], k: int) -> tuple[dict, list[float]]:
        recalls: list[float] = []
        ndcgs: list[float] = []
        mrrs: list[float] = []
        diversities: list[float] = []
        novelties: list[float] = []
        hhis: list[float] = []
        catalog: set[int] = set()
        for user_id, slate in slates.items():
            if not slate:
                continue
            truth = positives[user_id]
            hits = [int(item_pos in truth) for item_pos in slate[:k]]
            recalls.append(sum(hits) / max(1, min(len(truth), k)))
            dcg = sum(hit / log2(rank + 2) for rank, hit in enumerate(hits))
            ideal = sum(1 / log2(rank + 2) for rank in range(min(len(truth), k)))
            ndcgs.append(dcg / max(ideal, 1e-12))
            first_hit = next((rank + 1 for rank, hit in enumerate(hits) if hit), None)
            mrrs.append(0.0 if first_hit is None else 1.0 / first_hit)
            catalog.update(slate[:k])
            if len(slate) > 1:
                similarity = self.content_similarity[np.ix_(slate[:k], slate[:k])]
                upper = similarity[np.triu_indices_from(similarity, k=1)]
                diversities.append(float(1.0 - upper.mean()))
            novelties.append(float(np.mean([-log2(self.item_probability[item_pos]) for item_pos in slate[:k]])))
            creator_counts = Counter(self.items[item_pos]["creator"] for item_pos in slate[:k])
            hhis.append(sum((count / len(slate[:k])) ** 2 for count in creator_counts.values()))

        return (
            {
                "recall_at_10": round(float(np.mean(recalls)), 4),
                "ndcg_at_10": round(float(np.mean(ndcgs)), 4),
                "mrr_at_10": round(float(np.mean(mrrs)), 4),
                "catalog_coverage": round(len(catalog) / len(self.items), 4),
                "intra_list_diversity": round(float(np.mean(diversities)), 4),
                "novelty_bits": round(float(np.mean(novelties)), 4),
                "creator_hhi": round(float(np.mean(hhis)), 4),
            },
            ndcgs,
        )

    def evaluate(self, k: int = 10) -> dict:
        positives: dict[str, set[int]] = defaultdict(set)
        for event in self.evaluation_events:
            if event["action"] in {"save", "complete"}:
                positives[event["user_id"]].add(self.item_index[event["item_id"]])

        eligible = {user_id: truth for user_id, truth in positives.items() if truth}
        popularity_order = list(np.argsort(self.popularity)[::-1])
        slates: dict[str, dict[str, list[int]]] = {
            "popularity": {},
            "content": {},
            "accuracy": {},
            "balanced": {},
            "discovery": {},
        }
        for user_id in eligible:
            signals, sources = self._retrieve(user_id)
            seen = signals["seen"]
            slates["popularity"][user_id] = [pos for pos in popularity_order if self.items[pos]["item_id"] not in seen][:k]
            slates["content"][user_id] = [int(pos) for pos in np.argsort(signals["content"])[::-1] if self.items[int(pos)]["item_id"] not in seen][:k]
            for policy in ("accuracy", "balanced", "discovery"):
                recommendations = self.recommend(user_id, policy=policy, limit=k)["recommendations"]
                slates[policy][user_id] = [self.item_index[row["item_id"]] for row in recommendations]

        results: dict[str, dict] = {}
        per_user_ndcg: dict[str, list[float]] = {}
        for policy, policy_slates in slates.items():
            results[policy], per_user_ndcg[policy] = self._slate_metrics(policy_slates, eligible, k)

        rng = np.random.default_rng(self.seed)
        deltas = np.asarray(per_user_ndcg["balanced"]) - np.asarray(per_user_ndcg["popularity"])
        bootstrap = np.asarray([
            rng.choice(deltas, size=len(deltas), replace=True).mean() for _ in range(500)
        ])
        lower, upper = np.quantile(bootstrap, [0.025, 0.975])
        challenger, reference = results["balanced"], results["popularity"]
        guardrails = [
            {
                "id": "ndcg_gain_at_least_2pp",
                "claim": "Ranking gain over the popularity baseline is at least 2 points of NDCG@10.",
                "threshold": 0.02,
                "observed": round(challenger["ndcg_at_10"] - reference["ndcg_at_10"], 4),
                "passed": challenger["ndcg_at_10"] - reference["ndcg_at_10"] >= 0.02,
            },
            {
                "id": "coverage_not_lower_than_popularity",
                "claim": "The challenger surfaces at least as much of the catalog as the baseline.",
                "threshold": reference["catalog_coverage"],
                "observed": challenger["catalog_coverage"],
                "passed": challenger["catalog_coverage"] >= reference["catalog_coverage"],
            },
            {
                "id": "diversity_not_lower_than_popularity",
                "claim": "Slates are at least as varied as the baseline's.",
                "threshold": reference["intra_list_diversity"],
                "observed": challenger["intra_list_diversity"],
                "passed": challenger["intra_list_diversity"] >= reference["intra_list_diversity"],
            },
            {
                "id": "creator_hhi_at_most_0_22",
                "claim": "No slate concentrates exposure on a few creators.",
                "threshold": 0.22,
                "observed": challenger["creator_hhi"],
                "passed": challenger["creator_hhi"] <= 0.22,
            },
        ]
        return {
            "dataset": {
                "kind": "deterministic synthetic implicit-feedback simulation",
                "users": len(self.users),
                "items": len(self.items),
                "events": len(self.events),
                "eligible_evaluation_users": len(eligible),
                "seed": self.seed,
            },
            "training": self.training_summary,
            "policies": results,
            "paired_bootstrap": {
                "metric": "balanced NDCG@10 minus popularity NDCG@10",
                "mean_delta": round(float(deltas.mean()), 4),
                "confidence_interval_95": [round(float(lower), 4), round(float(upper), 4)],
                "resamples": 500,
            },
            "release_gate": {
                "status": "promote_to_shadow" if all(rule["passed"] for rule in guardrails) else "hold_for_review",
                "challenger": "balanced",
                "reference": "popularity",
                "guardrails": guardrails,
                "note": "Offline simulation supports shadow evaluation only; it is not causal online evidence.",
            },
        }

    def system_summary(self) -> dict:
        return {
            "name": "SignalWeave",
            "version": "0.1.0",
            "model_version": self.model_version,
            "serving": {"champion": CHAMPION_POLICY, "shadow": SHADOW_POLICY},
            "stages": [
                {"name": "Event contract", "detail": "time-ordered implicit actions, deduplicated per user and item"},
                {"name": "Hybrid retrieval", "detail": "content, item-item CF, time-decayed trending, exploration"},
                {"name": "Learning-to-rank", "detail": "8 features, logistic score for ordering (not a calibrated probability)"},
                {"name": "Slate optimizer", "detail": "greedy utility over relevance, diversity, novelty, freshness, creator cap"},
                {"name": "Release gate", "detail": "Top-K quality plus ecosystem guardrails on the frozen window"},
            ],
            "policies": POLICIES,
        }
