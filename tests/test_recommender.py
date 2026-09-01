import asyncio

import httpx

import pytest

from signalweave import registry
from signalweave.api import create_app
from signalweave.monitoring import compare_slates, population_stability_index
from signalweave.recommender import CHAMPION_POLICY, SHADOW_POLICY, SignalWeave
from signalweave.validation import run_validation


@pytest.fixture(scope="module")
def engine():
    return SignalWeave(seed=42)


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODELS_ROOT", tmp_path / "models")
    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "models" / "registry.json")
    return tmp_path


def test_recommendations_respect_contract_and_creator_cap():
    engine = SignalWeave(seed=42)
    result = engine.recommend("U001", policy="balanced", limit=10)

    assert len(result["recommendations"]) == 10
    assert len({row["item_id"] for row in result["recommendations"]}) == 10
    assert max(
        sum(candidate["creator"] == row["creator"] for candidate in result["recommendations"])
        for row in result["recommendations"]
    ) <= 2
    assert all(row["sources"] and row["why"] for row in result["recommendations"])


def test_temporal_evaluation_has_health_guardrails():
    report = SignalWeave(seed=42).report

    assert report["dataset"]["eligible_evaluation_users"] >= 50
    assert 0 <= report["policies"]["balanced"]["ndcg_at_10"] <= 1
    assert report["policies"]["balanced"]["catalog_coverage"] >= report["policies"]["popularity"]["catalog_coverage"]
    assert "status" in report["release_gate"]


def test_api_feedback_changes_live_state(tmp_path, monkeypatch):
    monkeypatch.setattr("signalweave.api.RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr("signalweave.api.DB_PATH", tmp_path / "feedback.sqlite3")
    engine = SignalWeave(seed=42)

    async def scenario():
        transport = httpx.ASGITransport(app=create_app(engine))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/api/health")).json()["status"] == "ok"
            response = await client.post(
                "/api/feedback",
                json={"user_id": "U001", "item_id": "L080", "action": "save"},
            )
            assert response.status_code == 201
            assert response.json()["feedback_events"] == 1

    asyncio.run(scenario())


def test_slate_explanation_reconstructs_the_score():
    engine = SignalWeave(seed=42)

    for row in engine.recommend("U007", policy="discovery", limit=6)["recommendations"]:
        decomposed = sum(term["contribution"] for term in row["utility_terms"])
        assert abs(decomposed - row["score"]) < 1e-3
        assert row["decided_by"] in {"relevance", "diversity", "novelty", "freshness"}
        assert len(row["contributions"]) == 8


def test_every_served_policy_is_evaluated():
    report = SignalWeave(seed=42).report

    assert {"popularity", "content", "accuracy", "balanced", "discovery"} <= set(report["policies"])
    for rule in report["release_gate"]["guardrails"]:
        assert {"id", "claim", "threshold", "observed", "passed"} <= set(rule)


def test_validation_suite_has_no_failing_checks():
    report = run_validation(SignalWeave(seed=42))
    failing = [
        check["id"]
        for group in report["groups"]
        for check in group["checks"]
        if check["status"] == "fail"
    ]

    assert failing == [], f"failing checks: {failing}"
    assert report["summary"]["checks"] >= 25


def test_registry_round_trip_serves_an_identical_slate(engine, isolated_registry):
    manifest = registry.register(engine, note="test")
    bundle = registry.load(feature_names=tuple(engine.training_summary["features"]))

    assert bundle.version == manifest["version"]
    assert bundle.code_is_current
    assert registry.model_digest(bundle.ranker) == manifest["model_digest"]

    served = SignalWeave(seed=42, bundle=bundle)
    assert served.model_version == manifest["version"]
    # The loaded model reports the metrics it was registered with, not a recomputation.
    assert served.report is bundle.report

    original = [row["item_id"] for row in engine.recommend("U005", CHAMPION_POLICY, 8)["recommendations"]]
    replayed = [row["item_id"] for row in served.recommend("U005", CHAMPION_POLICY, 8)["recommendations"]]
    assert original == replayed


def test_registry_refuses_a_bundle_with_a_different_feature_contract(engine, isolated_registry):
    registry.register(engine, note="test")

    with pytest.raises(registry.IncompatibleBundle):
        registry.load(feature_names=("content affinity", "a feature that does not exist"))


def test_promote_moves_the_champion(engine, isolated_registry):
    first = registry.register(engine, note="first")
    second = registry.register(engine, note="second")
    assert registry.read_registry()["champion"] == second["version"]

    result = registry.promote(first["version"])
    assert result["previous"] == second["version"]
    assert registry.read_registry()["champion"] == first["version"]


def test_shadow_is_logged_but_never_served(tmp_path, monkeypatch, engine):
    monkeypatch.setattr("signalweave.api.RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr("signalweave.api.DB_PATH", tmp_path / "feedback.sqlite3")

    async def scenario():
        transport = httpx.ASGITransport(app=create_app(engine))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            payload = (await client.get("/api/recommendations?user_id=U001&limit=8")).json()
            assert payload["policy"] == CHAMPION_POLICY

            shadow = payload["shadow"]
            assert shadow["policy"] == SHADOW_POLICY
            assert 0.0 <= shadow["overlap"] <= 1.0
            # The divergence is reported; the shadow slate itself is not.
            assert "recommendations" not in shadow and "items" not in shadow

            operations = (await client.get("/api/operations")).json()
            assert operations["shadow"]["requests"] == 1
            assert operations["latency"]["routes"]

            # An operator switching policies is an experiment, not production traffic.
            await client.get(f"/api/recommendations?user_id=U001&policy={SHADOW_POLICY}&limit=8")
            assert (await client.get("/api/operations")).json()["shadow"]["requests"] == 1

    asyncio.run(scenario())


def test_slate_comparison_and_psi_behave_at_the_edges():
    rows = [{"item_id": f"L{index:03d}", "score": 0.5} for index in range(5)]
    identical = compare_slates(rows, rows)
    assert identical["overlap"] == 1.0
    assert identical["top1_agree"] == 1
    assert identical["mean_rank_shift"] == 0.0

    disjoint = compare_slates(rows, [{"item_id": "X", "score": 0.1}])
    assert disjoint["overlap"] == 0.0
    assert disjoint["top1_agree"] == 0

    assert population_stability_index({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == 0.0
    assert population_stability_index({"a": 0.9, "b": 0.1}, {"a": 0.1, "b": 0.9}) > 0.25
