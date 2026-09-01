# Operations

What separates this from a notebook is not the model. It is that the model is an
artifact with a name, produced by a pipeline that refuses to ship it unproven,
served by a process that cannot silently substitute a different one, and watched
by something that would notice if the traffic stopped resembling the training set.

Everything here runs on a laptop. There is no orchestrator, no metrics backend
and no feature store, because at 80 items and 2,176 events none of those would be
solving a measured problem. What is here is the part that would still be true at
a thousand times the scale.

## The loop

```text
python -m signalweave.train
    fit  →  evaluate on the frozen window  →  run 31 checks
                                                   │
                              any check `fail`  ───┴──→  refuse to register
                                                   │
                              all clear ───────────┴──→  write bundle + manifest
                                                          promote to champion

python -m signalweave                     load champion, serve, never fit
    │
    ├── every champion request also scores the shadow policy and logs the divergence
    ├── latency and errors accumulate per route
    └── live feedback accumulates in SQLite and is compared to the training mix

python -m signalweave.train --promote VERSION      roll forward or back
```

## Training is not serving

Before: the app fitted a ranker at import time. The metrics it displayed came
from a computation that happened to run the same code as the one answering
requests — which is nearly the same thing as them belonging to the same model,
and nearly is where incidents live.

Now `SignalWeave(bundle=...)` adopts a registered model: its scaler, its
coefficients, its fit matrix, and the frozen-window report it scored when it was
registered. The API constructs the engine this way and never calls `_fit_ranker`.

Two consequences worth stating:

- **Startup is a load.** Building retrieval state and adopting a bundle takes
  about 0.2 s against roughly 4 s to fit and evaluate.
- **The metrics on the Policies tab belong to the served model.** They are read
  out of the bundle. `serving.metrics_belong_to_the_served_model` checks that.

## The registry

```text
artifacts/models/
├── registry.json                         champion pointer + history
└── 20260831T140130Z-5455be7e/
    ├── manifest.json                     committed
    └── model.joblib                      gitignored
```

A version id is `<UTC timestamp>-<first 8 of the model digest>`: sortable, and it
names the coefficients it contains.

The manifest records what the binary cannot prove about itself:

| Field | Answers |
|---|---|
| `data_digest` | which dataset this was fitted on |
| `code_digest` | which revision of `data.py` + `recommender.py` defined the features |
| `model_digest` | which coefficients, so a swapped binary is detectable |
| `environment` | Python, numpy and scikit-learn versions |
| `training` | held-out ROC AUC, Brier, calibration gap, coefficients |
| `metrics` | frozen-window results for every policy |
| `validation` | the check tally at training time |
| `serving_baseline` | mean slate score, so drift has a reference |

The binary is gitignored and the manifest is not. A model you can rebuild is
cheap; a model you cannot account for is a liability.

Five checks in the `serving` group cover this at runtime, and they only appear
when a registered model is loaded:

- `model_came_from_the_registry` — not an in-process fit
- `binary_matches_its_manifest` — coefficients hash to what was recorded
- `metrics_belong_to_the_served_model` — the report came from the bundle
- `data_has_not_moved_under_the_model` — the generator still produces the fitted dataset
- `serving_code_matches_training_code` — `known`, not `fail`, when they diverge

That last one is deliberately not a failure. Serving a model trained by an older
revision is ordinary between deploys. What is not ordinary is *not knowing*.

## Registration is gated

`train.py` runs the full check suite before it writes anything, and a run that
produces any `fail` does not become an artifact:

```text
refusing to register: 1 failing check(s): artifact_matches_runtime
Fix them, or pass --allow-failing-checks to register a known-bad model deliberately.
```

The escape hatch exists because sometimes you do want to register a model you
know is wrong — to reproduce an incident, or to hold a baseline. It has to be
asked for.

A seed sweep is an experiment, not a release: runs on any seed other than 42 get
a registry entry but leave `artifacts/` alone, and `artifact_matches_runtime`
reports `known` rather than comparing them against a record they were never
meant to match.

## Shadow deployment

The release gate returns `promote_to_shadow`. This is where that verdict is acted
on rather than displayed.

- **Champion** is `accuracy`, the policy served by default. It leads every ranking
  metric on the frozen window and reaches the whole catalog.
- **Shadow** is `balanced`, the challenger. It is scored on every champion request
  and never returned.

Each comparison is logged to `shadow_log` in SQLite: overlap with the served
slate, whether the top pick agrees, the mean rank shift of shared items, and both
latencies. On a typical request the shadow policy keeps 7 of 8 items with the
same top pick — which is the useful finding. A challenger that reorders almost
nothing does not need an experiment to justify it, and one that reorders
everything needs a much more careful one.

Two deliberate limits:

- Only the divergence leaves the process. The shadow slate itself is never in the
  response, because a shadow slate that reaches a user is not a shadow.
- An operator switching policies in the console is an experiment, not traffic, so
  those requests are not logged.

The cost is honest: a champion request scores two slates. Both latencies are
recorded separately so the overhead is visible rather than absorbed.

## Monitoring

`GET /api/operations` returns everything the Operations tab shows.

**Latency** is an in-process ring buffer over the last 1,000 requests, per route,
reporting p50, p95, max and error count. It resets when the service restarts,
which is stated in the UI rather than glossed over.

**Action-mix drift** compares the live feedback distribution against the actions
the ranker was fitted on, as a population stability index. Below 30 live events
it reports `insufficient_data` instead of a number, because a PSI computed on
nine events is noise wearing a decimal point. Conventional bands apply above
that: under 0.10 stable, 0.10–0.25 watch, 0.25 and over drifted.

**Slate-score drift** compares the mean utility of served slates against
`serving_baseline.mean_slate_score` from the manifest. This is the monitor that
catches a silent failure — retrieval degrading, a feature going constant — with
no error and no exception.

**Retrain** counts feedback events since registration against a threshold. Live
feedback updates a learner's profile immediately, but it never moves the ranker
coefficients. Only a training run does that, and it produces a new version.

## CI

`.github/workflows/ci.yml` runs the tests, then the check suite, then the
training pipeline with `--no-promote`.

The check suite is the interesting one: it exits non-zero on any `fail`, and
`artifact_matches_runtime` fails when `artifacts/evaluation.json` no longer
matches what the code produces. Committing a change that moves the metrics
without regenerating the evidence breaks the build.

## What is deliberately absent

- **An orchestrator.** One machine, one command, seconds of runtime.
- **A metrics backend.** A ring buffer answers the question at this scale, and
  swapping it for a real exporter later is a small change at one boundary.
- **A feature store.** Features are computed from in-memory arrays in the same
  process that serves them, so there is no training/serving skew surface to
  guard — which is itself the reason none is needed yet.
- **Automatic retraining.** The retrain signal is surfaced; pulling the trigger
  stays a decision, because retraining on feedback the current policy generated
  is how a recommender teaches itself its own habits.
