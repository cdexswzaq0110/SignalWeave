# API Contract

Base URL: `http://127.0.0.1:8010`

## Health

`GET /api/health`

```json
{"status":"ok","model_ready":true,"feedback_events":0}
```

## Users

`GET /api/users`

Returns demo learner profiles including intent topics, preferred format, level, and time budget.

## Recommendations

`GET /api/recommendations?user_id=U001&policy=balanced&limit=8`

Validation:

- `user_id` must exist.
- `policy` is `accuracy`, `balanced`, or `discovery`. It defaults to the champion policy.
- `limit` is from 1 through 20.

When the requested policy is the champion, the shadow policy is scored behind it and the
comparison is logged. The response then carries a `shadow` block with `overlap`, `top1_agree`,
`mean_rank_shift` and both latencies. The shadow slate itself is never included — only how far it
diverged.

Response fields include learner context, active policy weights, catalog size, how many items were
excluded as already seen, the candidate count and source funnel, and the ranked items.

Each item carries its full decision record:

| Field | Meaning |
|---|---|
| `score` | final slate utility that placed it at this rank |
| `utility_terms` | the four weighted terms — `weight`, raw `value`, `contribution`, `share` — summing to `score` |
| `decided_by` | the term with the largest contribution |
| `relevance` | ranker score (ordering only; not a calibrated probability) |
| `contributions` | all eight features as `coefficient × standardized value`, summing to the logit |
| `sources` | which retrieval routes proposed it |
| `runner_up` | the candidate it beat at this position, and the margin |
| `blocked_by_creator_cap` | candidates the cap removed from consideration at this step |

Errors return HTTP 400:

```json
{"detail":"Unknown policy: invalid"}
```

## Policy comparison

`GET /api/compare?user_id=U001&limit=8`

Returns accuracy, balanced, and discovery slates for the same learner.

## Feedback

`POST /api/feedback`

```json
{"user_id":"U001","item_id":"L080","action":"save"}
```

Allowed actions: `complete`, `save`, `open`, `dismiss`.

The validated event is appended to `runtime/feedback.sqlite3`, added to the in-memory learner context, and acknowledged with HTTP 201. SQL parameters are bound rather than interpolated.

## Evaluation

`GET /api/evaluation`

Returns dataset facts, the split contract, the ranker training summary (held-out ROC AUC, Brier,
calibration gap, coefficients), per-policy metrics for both baselines and all three served
policies, paired-bootstrap uncertainty, and the release gate.

`release_gate.guardrails` is a list; each entry carries `id`, `claim`, `threshold`, `observed`
and `passed`, so the gate can be read without knowing the thresholds in advance.

## Validation

`GET /api/validation`

Re-runs the full check suite against the live process and returns groups of checks, each with
`claim`, `expected`, `observed`, `status` (`pass` / `fail` / `known`) and an optional `detail`.

Computed on first request and cached for the process lifetime, because it re-runs retrieval for a
sample of learners. The same report is available offline as `artifacts/validation.json` and from
`python -m signalweave.validation`.

## Model and operations

`GET /api/model`

Which version is loaded, where it came from, whether the serving code still matches the code that
trained it, and every registered manifest.

`GET /api/metrics`

Per-route request counts and p50 / p95 / max latency over an in-process window of the last 1,000
requests. Resets on restart.

`GET /api/operations`

One round trip for the operations view: `model`, `serving` (champion and shadow policies),
`latency`, `shadow` (accumulated divergence) and `drift` (action mix PSI, slate-score drift,
retrain signal).

## System metadata

`GET /api/system`

Returns stage descriptions and policy weights used by the interface.

Interactive OpenAPI documentation is available at `/docs`.

