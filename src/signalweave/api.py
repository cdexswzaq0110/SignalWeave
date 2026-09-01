"""FastAPI boundary: serving, feedback persistence, and the operational surface.

Serving loads a registered model. It does not fit one on the request path, and it
does not recompute the metrics it reports — those travelled with the model. What
this layer adds on top is the part only a running service can know: latency, how
far the shadow policy diverges from what was served, and whether live traffic
still resembles what the champion was trained on.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import registry
from .monitoring import ServingMetrics, compare_slates, drift_report, shadow_summary
from .recommender import CHAMPION_POLICY, POLICIES, SHADOW_POLICY, SignalWeave
from .train import load_or_bootstrap
from .validation import run_validation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = PROJECT_ROOT / "web"
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
DB_PATH = RUNTIME_ROOT / "feedback.sqlite3"

SHADOW_WINDOW = 500


class FeedbackRequest(BaseModel):
    user_id: str
    item_id: str
    action: str


def _connect() -> sqlite3.Connection:
    RUNTIME_ROOT.mkdir(exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            action TEXT NOT NULL,
            occurred_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS shadow_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            occurred_at TEXT NOT NULL,
            model_version TEXT NOT NULL,
            user_id TEXT NOT NULL,
            champion_policy TEXT NOT NULL,
            shadow_policy TEXT NOT NULL,
            k INTEGER NOT NULL,
            overlap REAL NOT NULL,
            top1_agree INTEGER NOT NULL,
            mean_rank_shift REAL NOT NULL,
            champion_mean_score REAL NOT NULL,
            champion_ms REAL NOT NULL,
            shadow_ms REAL NOT NULL
        )
        """
    )
    return connection


def _read_shadow(limit: int = SHADOW_WINDOW) -> list[dict]:
    columns = (
        "occurred_at", "model_version", "user_id", "champion_policy", "shadow_policy",
        "k", "overlap", "top1_agree", "mean_rank_shift", "champion_mean_score",
        "champion_ms", "shadow_ms",
    )
    with _connect() as connection:
        rows = connection.execute(
            f"SELECT {', '.join(columns)} FROM shadow_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(zip(columns, row)) for row in rows]


def create_app(engine: SignalWeave | None = None) -> FastAPI:
    app = FastAPI(title="SignalWeave API", version="0.1.0")
    recommender = engine or load_or_bootstrap()
    metrics = ServingMetrics()
    # The check suite re-runs retrieval for a sample of learners, so it is computed
    # on first request rather than at import time.
    validation_cache: dict[str, dict | None] = {"report": None}

    @app.middleware("http")
    async def record_latency(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            metrics.record(request.url.path, (time.perf_counter() - started) * 1000, response.status_code)
        return response

    with _connect() as connection:
        rows = connection.execute("SELECT user_id, item_id, action FROM feedback ORDER BY id").fetchall()
    for user_id, item_id, action in rows:
        try:
            recommender.record_feedback(user_id, item_id, action)
        except (KeyError, ValueError):
            continue

    def _manifest() -> dict:
        return recommender.bundle.manifest if recommender.bundle is not None else {}

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model_version": recommender.model_version,
            "provenance": "registry" if recommender.bundle is not None else "unregistered",
            "code_is_current": recommender.bundle.code_is_current if recommender.bundle else True,
            "feedback_events": len(recommender.live_feedback),
        }

    @app.get("/api/users")
    def users() -> list[dict]:
        return recommender.users

    @app.get("/api/system")
    def system() -> dict:
        return recommender.system_summary()

    @app.get("/api/evaluation")
    def evaluation() -> dict:
        return recommender.report

    @app.get("/api/validation")
    def validation() -> dict:
        """Re-run every data and model check against this live process."""

        if validation_cache.get("report") is None:
            validation_cache["report"] = run_validation(recommender)
        return validation_cache["report"]

    @app.get("/api/model")
    def model() -> dict:
        """Which model is serving, where it came from, and what else is registered."""

        return registry.registry_state(recommender.bundle)

    @app.get("/api/metrics")
    def serving_metrics() -> dict:
        return metrics.snapshot()

    @app.get("/api/operations")
    def operations() -> dict:
        """Everything the operations view needs, in one round trip."""

        shadow_rows = _read_shadow()
        with _connect() as connection:
            feedback_rows = connection.execute("SELECT user_id, item_id, action FROM feedback").fetchall()
        return {
            "model": registry.registry_state(recommender.bundle),
            "serving": {"champion_policy": CHAMPION_POLICY, "shadow_policy": SHADOW_POLICY},
            "latency": metrics.snapshot(),
            "shadow": shadow_summary(shadow_rows, CHAMPION_POLICY, SHADOW_POLICY),
            "drift": drift_report(
                recommender,
                list(feedback_rows),
                baseline=_manifest().get("serving_baseline"),
                shadow_rows=shadow_rows,
            ),
        }

    @app.get("/api/recommendations")
    def recommendations(
        user_id: str,
        policy: str = Query(default=CHAMPION_POLICY),
        limit: int = Query(default=8, ge=1, le=20),
    ) -> dict:
        try:
            started = time.perf_counter()
            result = recommender.recommend(user_id, policy=policy, limit=limit)
            champion_ms = (time.perf_counter() - started) * 1000
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        # Shadow runs only behind production traffic. An operator switching policies
        # in the console is an experiment, not traffic, so it is not logged.
        if policy == CHAMPION_POLICY and SHADOW_POLICY != CHAMPION_POLICY:
            started = time.perf_counter()
            shadow = recommender.recommend(user_id, policy=SHADOW_POLICY, limit=limit)
            shadow_ms = (time.perf_counter() - started) * 1000
            comparison = compare_slates(result["recommendations"], shadow["recommendations"])
            occurred_at = datetime.now(UTC).isoformat()
            with _connect() as connection:
                connection.execute(
                    """
                    INSERT INTO shadow_log (
                        occurred_at, model_version, user_id, champion_policy, shadow_policy,
                        k, overlap, top1_agree, mean_rank_shift, champion_mean_score,
                        champion_ms, shadow_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        occurred_at, recommender.model_version, user_id, policy, SHADOW_POLICY,
                        comparison["k"], comparison["overlap"], comparison["top1_agree"],
                        comparison["mean_rank_shift"], comparison["champion_mean_score"],
                        round(champion_ms, 3), round(shadow_ms, 3),
                    ),
                )
            # Only the divergence is returned. The shadow slate itself is never served,
            # which is the whole point of running it in shadow.
            result["shadow"] = {
                **comparison,
                "policy": SHADOW_POLICY,
                "champion_ms": round(champion_ms, 2),
                "shadow_ms": round(shadow_ms, 2),
            }

        result["model_version"] = recommender.model_version
        return result

    @app.get("/api/compare")
    def compare(user_id: str, limit: int = Query(default=8, ge=1, le=20)) -> dict:
        try:
            return {
                policy: recommender.recommend(user_id, policy=policy, limit=limit)
                for policy in POLICIES
            }
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/feedback", status_code=201)
    def feedback(payload: FeedbackRequest) -> dict:
        try:
            recommender.record_feedback(payload.user_id, payload.item_id, payload.action)
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        occurred_at = datetime.now(UTC).isoformat()
        with _connect() as connection:
            connection.execute(
                "INSERT INTO feedback (user_id, item_id, action, occurred_at) VALUES (?, ?, ?, ?)",
                (payload.user_id, payload.item_id, payload.action, occurred_at),
            )
        return {"accepted": True, "occurred_at": occurred_at, "feedback_events": len(recommender.live_feedback)}

    @app.get("/api/export/evaluation")
    def export_evaluation() -> dict:
        return json.loads(json.dumps(recommender.report))

    app.mount("/assets", StaticFiles(directory=WEB_ROOT), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html")

    return app


app = create_app()
