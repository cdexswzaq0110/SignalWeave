# SignalWeave Model Card

## Intended use

SignalWeave demonstrates a complete recommendation decision path for technical interviews and local learning. It is suitable for inspecting retrieval, ranking, re-ranking, explanations, offline Top-K metrics, and release gates.

It is not suitable for production decisions about real learners or creators without representative data, privacy review, online instrumentation, and controlled experimentation.

## Data

- 64 synthetic learner profiles.
- 80 synthetic learning-content items across 8 topics and 20 creators.
- 2,176 deterministic implicit events: complete, save, open, dismiss.
- Generator seed: 42.
- Per-user chronological split: 60% representation history, 20% ranker fit, 20% frozen evaluation.

Synthetic generation intentionally contains preference structure so contracts and metrics are testable. This also means performance is not externally valid.

## Model and policies

- Retrieval: TF-IDF content, item-item collaborative, time-decayed trending (21-day half-life), exploration.
- Ranker: standardized class-balanced logistic regression. Held-out 5-fold ROC AUC 0.6402, Brier 0.2254.
- Re-ranker: greedy multi-objective slate optimization with creator cap.
- Cold-start fallback: trending and exploration routes remain available when personalized signals are weak.

## Evaluation

Positive frozen labels are `save` and `complete`. Metrics are averaged across users with at least one positive frozen event.

Accuracy metrics:

- Recall@10
- NDCG@10
- MRR@10

Slate and ecosystem metrics:

- catalog coverage
- mean pairwise content diversity
- self-information novelty in bits
- creator Herfindahl-Hirschman Index (lower is less concentrated)

All three served policies are scored, not only the challenger. On the frozen window the accuracy
policy leads on NDCG@10 (0.2540), ahead of the content baseline (0.2354) and the balanced
challenger (0.2131).

Retrieval reaches 0.9111 of frozen-window positives, which caps every ranking metric above.

A paired user bootstrap quantifies sampling uncertainty in the NDCG@10 delta. It does not correct
exposure bias and is not a causal estimate.

## Known limitations

Four of these are asserted continuously by the check suite rather than only described here:
`features_vary`, `features_carry_weight`, `score_is_a_probability` and
`challenger_beats_strongest_baseline` all return `known` on every run.

1. The event simulator is structurally simpler than real user behavior. Every learner has exactly
   34 events, so sparse-user and repeat-exposure behavior is untested.
2. Missing-not-at-random exposure bias is not estimated with propensity scores.
3. Pointwise negative sampling treats unobserved items as negatives during ranker fit.
4. Text features use catalog metadata, not semantic embeddings.
5. Live feedback updates the user profile but does not retrain global ranker coefficients.
6. Creator HHI is a slate diagnostic, not a complete fairness definition.
7. The greedy optimizer is not guaranteed to find the global optimum.
8. The ranker score is not calibrated: `class_weight="balanced"` puts its mean at 0.4706 against an
   observed positive rate of 0.2340. Use it for ordering, never for expected-value arithmetic.
9. Two of the eight features (`editorial quality`, `popularity`) carry coefficients near zero on
   this catalog. They are retained because they are cheap and would matter on data with real
   variance, but they are not doing work here.
10. The release gate compares the challenger against the popularity baseline only. Balanced trails
    the stronger content baseline by 0.0223 NDCG@10, and no guardrail tests for that.

## Production path

Before real deployment:

1. Establish consent, retention, deletion, and sensitive-feature policies.
2. Train on representative logged impressions, not only positive interactions.
3. Add propensity-aware offline evaluation or randomized exploration data.
4. Run shadow traffic with latency, freshness, and feature-parity monitoring.
5. Define primary and guardrail metrics with product and marketplace stakeholders.
6. Run a pre-registered online experiment; monitor segment and creator-level harms.
7. Promote only when both user-value and ecosystem guardrails pass.

