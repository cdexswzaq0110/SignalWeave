# System Architecture

## Data ownership and lifecycle

| Data | Owner | Mutation | Lifetime |
|---|---|---|---|
| Catalog and demo users | deterministic generator | immutable for seed 42 | process lifetime |
| Historical implicit events | deterministic generator | immutable | process lifetime |
| Retrieval indexes | recommender engine | rebuilt at startup | process lifetime |
| Ranker coefficients | training pipeline | fitted by `python -m signalweave.train` | registered version |
| Live feedback | API boundary | append-only | SQLite + memory |
| Evaluation report | recommender engine | rebuilt at startup | process lifetime |
| Validation report | check suite | computed on first request, then cached | process lifetime |
| Ranker coefficients (served) | model registry | immutable per version | on disk until promoted away |
| Shadow comparisons | API boundary | append-only | SQLite |
| Latency window | serving metrics | ring buffer, last 1000 requests | process lifetime |

## Data flow

```text
catalog + users + implicit events
→ per-user chronological split
→ history-only representations and retrieval indexes
→ ranker-fit examples and sampled negatives
→ frozen per-user Top-K evaluation
→ policy metrics and release gate
→ FastAPI responses
→ browser feedback
→ validated SQLite append
→ next-request profile update
```

## Candidate generation

Each route emits its top 22 unseen items. The union is deduplicated while retaining source attribution.

- **Content:** cosine similarity between the learner's weighted TF-IDF profile and item text.
- **Collaborative:** item-item similarity derived from the history-only implicit user-item matrix.
- **Trending:** positive event weight decayed by event age with a 21-day half-life, measured
  against the last timestamp in the history window.

  An earlier build weighted by catalog position instead of event time, which made trending 0.99
  correlated with popularity — a fourth route that was really a copy of a feature the ranker
  already had. `retrieval.sources_are_not_redundant` in the check suite is what now holds that
  down; the same pair currently measures 0.90.
- **Exploration:** quality and freshness with an inverse-popularity bonus.

## Ranker

A class-balanced logistic regression scores candidates from eight `[0,1]` features:

1. content affinity
2. collaborative affinity
3. topic affinity
4. popularity
5. freshness
6. editorial quality
7. difficulty fit
8. duration fit

Features are standardized before fitting.

The same recipe is scored with 5-fold cross-validation inside the ranker-fit window: **ROC AUC
0.6402**, **Brier 0.2254** over 842 rows. The frozen window is never touched by this.

`class_weight="balanced"` reweights the classes, so the sigmoid output is deliberately *not* a
calibrated probability — its mean (0.4706) sits well above the observed positive rate (0.2340).
It is used for ordering only, and the check suite records the gap rather than implying it away.

Per-item explanations use `coefficient × standardized feature value`. Those contributions sum
back to the logit that produced the score, which `model.explanations_reconstruct_the_score`
verifies to within 1e-15: the numbers in the UI are the numbers the model used, not a
re-derivation.

## Slate optimization

The ranker orders individual candidates. A greedy slate optimizer then maximizes:

```text
policy utility =
  relevance_weight × ranker_probability
+ diversity_weight × distance_from_selected_items
+ novelty_weight × inverse_popularity
+ freshness_weight × recency
```

A hard creator cap is checked before utility comparison. This keeps a business/product constraint
separate from the model score.

Each selected item carries its own decomposition out through the API: the four weighted terms with
their contributions and shares, the term that decided the position, the runner-up candidate and the
margin it lost by, and how many candidates the creator cap removed at that step.

## Validation

`src/signalweave/validation.py` re-derives every claim against the live process — 31 checks across
dataset, split, features, model, retrieval and evaluation. It is exposed at `GET /api/validation`,
rendered in the Checks tab of the console, runnable as `python -m signalweave.validation`, and
asserted by the test suite.

Two of its checks exist specifically to catch the failure modes that offline recsys work is prone
to: `split.chronological_per_user` would catch training on the future, and
`retrieval.candidate_recall_ceiling` reports the fraction of frozen-window positives that retrieval
can reach at all (currently 0.9111), which is the hard ceiling on any ranking metric downstream.

## Failure handling

- Unknown users, items, actions, policies, or invalid limits return HTTP 400.
- Static files and APIs share one origin; no permissive CORS policy is required.
- SQLite writes are parameterized and append-only.
- Invalid historical feedback rows are skipped at startup rather than breaking the model.
- The UI renders request errors in its active panel and keeps controls usable.

## Training and serving

Training and serving are separate processes with a versioned artifact between them. `train.py`
fits, evaluates on the frozen window, runs the check suite, and registers a bundle only if nothing
returns `fail`. The API constructs `SignalWeave(bundle=...)`, which adopts the stored scaler,
coefficients, fit matrix and frozen-window report instead of fitting anything.

This is what makes the metrics on screen attributable: they are read from the bundle the served
model was registered with, not recomputed by code that happens to be the same. See
[mlops.md](mlops.md) for the registry layout, the shadow deployment, and the drift monitors.

## Deployment boundary

This build is a single-machine portfolio system. A measured need for independent scaling would justify separating training, retrieval index generation, online serving, and event ingestion. Those services are intentionally absent from the MVP.

