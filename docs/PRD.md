# SignalWeave — Product Requirements

## Product statement

SignalWeave is a local recommendation decision console for a learning-content marketplace. It helps an ML/product team inspect how implicit feedback becomes a Top-K slate and decide whether a ranking policy is safe enough for shadow evaluation.

## Target users

- ML engineer validating retrieval, ranking, and temporal evaluation.
- Product scientist comparing relevance with discovery and ecosystem health.
- Marketplace operator checking why an item was recommended and whether a creator cap was applied.

## Core jobs

1. Generate a personalized slate from time-ordered implicit events.
2. Compare accuracy, balanced, and discovery policies for the same learner.
3. Explain each item through ranker signals and candidate sources.
4. Measure both ranking quality and whole-slate health.
5. Persist new feedback and reflect it on the next recommendation request.
6. Make a bounded release decision with visible offline guardrails.
7. Re-verify the data and model claims on every run instead of trusting the documentation.
8. Ship a named model version that can be rolled back, and watch it once it is serving.

## MVP acceptance criteria

- At least three distinct candidate retrieval sources are combined and deduplicated.
- Previously seen items are excluded from the recommended slate.
- Ranker training and evaluation use distinct chronological windows.
- API returns exactly the requested `1..20` recommendations when enough candidates exist.
- Balanced policy enforces no more than two items per creator.
- Evaluation reports Recall@10, NDCG@10, MRR@10, catalog coverage, intra-list diversity, novelty, and creator HHI.
- Paired user bootstrap reports a 95% interval for NDCG delta.
- UI supports policy switching, learner switching, feedback, and all three analysis views.
- Every policy the API can serve has frozen-window metrics of its own.
- Each recommended item exposes a decomposition that reconstructs its own score.
- The check suite runs clean: no check returns `fail`, and every unmet claim is recorded as a
  named `known` limitation rather than omitted.
- Serving loads a registered model rather than fitting one on the request path.
- A training run with any failing check does not register.
- The challenger is scored on every champion request without ever being served.
- Local tests and browser console complete without errors.

## Non-goals

- No claim of real-world CTR, retention, or revenue uplift.
- No production authentication, distributed serving, Kafka, feature store, or vector database.
- No orchestrator, metrics backend, or automatic retraining. The retrain signal is surfaced;
  pulling the trigger stays a decision.
- No deep ranker until real data volume and latency evidence justify one.
- No online A/B test emulation presented as causal evidence.

## Success definition

The MVP is complete when one command starts the app, deterministic tests pass, every surface loads from the API, user feedback is persisted locally, and the model card names the simulation boundary without ambiguity.

