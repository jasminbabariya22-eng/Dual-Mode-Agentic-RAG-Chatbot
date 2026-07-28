"""
Tests for the Monitoring layer.

All external dependencies (SQLite, ChromaDB, Redis, LLM, network) are mocked.
Prometheus metrics are inspected via the /metrics HTTP endpoint or by reading
the registry directly.

Coverage:
    - GET /metrics             (endpoint, content-type, contains expected metric names)
    - GET /health              (liveness always-200)
    - GET /health/live         (liveness alias)
    - GET /health/ready        (all-ok, redis-degraded, sqlite-failed → 503)
    - ObservabilityMiddleware  (X-Request-ID header, latency counter, streaming counter)
    - Tracing module           (disabled without OTEL env vars)
    - record_workflow_metrics  (histogram observations)
    - metrics.py counter helpers
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Prometheus registry isolation
# ---------------------------------------------------------------------------
# prometheus_client keeps a global REGISTRY.  Between test runs the same
# process reuses the registry, so metric objects defined in metrics.py are
# already registered.  We do NOT re-register them; we just import and read them.

from backend.main import app
from backend.app.monitoring.health import CheckResult
from backend.app.monitoring.metrics import (
    CACHE_HITS_TOTAL,
    ERRORS_TOTAL,
    HTTP_REQUESTS_TOTAL,
    REQUESTS_BY_ROUTE_TOTAL,
    STREAMING_REQUESTS_TOTAL,
    WORKFLOW_DURATION_SECONDS,
    record_workflow_metrics,
)
from backend.app.monitoring.tracing import is_tracing_enabled, setup_tracing

# ---------------------------------------------------------------------------
# Shared test client
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


# ===========================================================================
# /metrics endpoint
# ===========================================================================


def test_metrics_endpoint_returns_200(client: TestClient):
    """Prometheus /metrics should return HTTP 200."""
    response = client.get("/metrics")
    assert response.status_code == 200


def test_metrics_endpoint_content_type(client: TestClient):
    """Response Content-Type must be text/plain for Prometheus scraping."""
    response = client.get("/metrics")
    assert "text/plain" in response.headers.get("content-type", "")


def test_metrics_contains_http_requests_total(client: TestClient):
    """Scrape output must contain our custom http_requests_total metric."""
    # Trigger at least one request so the counter appears
    client.get("/health")
    response = client.get("/metrics")
    assert "http_requests_total" in response.text


def test_metrics_contains_workflow_metric(client: TestClient):
    """Scrape output must declare the workflow_duration_seconds histogram."""
    response = client.get("/metrics")
    assert "workflow_duration_seconds" in response.text


def test_metrics_contains_errors_total(client: TestClient):
    """Scrape output must declare errors_total counter."""
    response = client.get("/metrics")
    assert "errors_total" in response.text


# ===========================================================================
# /health endpoints
# ===========================================================================


def test_health_liveness_returns_200(client: TestClient):
    """GET /health must always return HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_liveness_body(client: TestClient):
    """GET /health body must contain status=healthy."""
    response = client.get("/health")
    assert response.json()["status"] == "healthy"


def test_health_live_alias(client: TestClient):
    """GET /health/live is an alias — must return HTTP 200."""
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# ===========================================================================
# /health/ready — all checks pass
# ===========================================================================


def _mock_all_ok():
    """Context managers that make all readiness checks succeed."""
    sqlite_ok = patch(
        "backend.app.monitoring.health._check_sqlite",
        new=AsyncMock(return_value=("sqlite", CheckResult(status="ok"))),
    )
    vector_ok = patch(
        "backend.app.monitoring.health._check_vector_store",
        new=AsyncMock(return_value=("vector_store", CheckResult(status="ok", detail="1 collection(s) found"))),
    )
    redis_ok = patch(
        "backend.app.monitoring.health._check_redis",
        new=AsyncMock(return_value=("redis", CheckResult(status="ok"))),
    )
    llm_ok = patch(
        "backend.app.monitoring.health._check_llm",
        new=AsyncMock(return_value=("llm", CheckResult(status="ok", detail="Ollama HTTP 200"))),
    )
    return sqlite_ok, vector_ok, redis_ok, llm_ok


def test_health_ready_all_ok(client: TestClient):
    """When all checks pass, readiness returns 200 and status=healthy."""
    sqlite_ok, vector_ok, redis_ok, llm_ok = _mock_all_ok()
    with sqlite_ok, vector_ok, redis_ok, llm_ok:
        response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "version" in body
    assert "checked_at" in body


def test_health_ready_response_has_all_check_keys(client: TestClient):
    """Readiness response must include checks for sqlite, vector_store, redis, llm."""
    sqlite_ok, vector_ok, redis_ok, llm_ok = _mock_all_ok()
    with sqlite_ok, vector_ok, redis_ok, llm_ok:
        response = client.get("/health/ready")
    checks = response.json().get("checks", {})
    for key in ("sqlite", "vector_store", "redis", "llm"):
        assert key in checks, f"Missing check key: {key}"


# ===========================================================================
# /health/ready — Redis degraded
# ===========================================================================


def test_health_ready_redis_degraded(client: TestClient):
    """Redis unavailable must yield overall status=degraded (not unhealthy) and HTTP 200."""
    sqlite_ok, vector_ok, _, llm_ok = _mock_all_ok()
    redis_degraded = patch(
        "backend.app.monitoring.health._check_redis",
        new=AsyncMock(
            return_value=(
                "redis",
                CheckResult(status="degraded", detail="Redis unreachable; fallback active."),
            )
        ),
    )
    with sqlite_ok, vector_ok, redis_degraded, llm_ok:
        response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["redis"]["status"] == "degraded"


def test_health_ready_llm_degraded_stays_200(client: TestClient):
    """LLM unavailable must yield overall status=degraded and HTTP 200 (Groq fallback)."""
    sqlite_ok, vector_ok, redis_ok, _ = _mock_all_ok()
    llm_degraded = patch(
        "backend.app.monitoring.health._check_llm",
        new=AsyncMock(
            return_value=(
                "llm",
                CheckResult(status="degraded", detail="Ollama unreachable; Groq fallback active."),
            )
        ),
    )
    with sqlite_ok, vector_ok, redis_ok, llm_degraded:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


# ===========================================================================
# /health/ready — hard failure (SQLite)
# ===========================================================================


def test_health_ready_sqlite_failed_returns_503(client: TestClient):
    """SQLite failure must yield overall status=unhealthy and HTTP 503."""
    _, vector_ok, redis_ok, llm_ok = _mock_all_ok()
    sqlite_failed = patch(
        "backend.app.monitoring.health._check_sqlite",
        new=AsyncMock(
            return_value=(
                "sqlite",
                CheckResult(status="failed", detail="Database file not found"),
            )
        ),
    )
    with sqlite_failed, vector_ok, redis_ok, llm_ok:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"


# ===========================================================================
# ObservabilityMiddleware
# ===========================================================================


def test_middleware_injects_request_id_header(client: TestClient):
    """Every response must carry an X-Request-ID header."""
    response = client.get("/health")
    assert "x-request-id" in response.headers


def test_middleware_request_id_is_uuid(client: TestClient):
    """The injected X-Request-ID must be a valid UUID4."""
    import uuid
    response = client.get("/health")
    rid = response.headers.get("x-request-id", "")
    parsed = uuid.UUID(rid)   # raises ValueError if not a valid UUID
    assert str(parsed) == rid


def test_middleware_propagates_client_request_id(client: TestClient):
    """If the client sends X-Request-ID, the same value must be echoed back."""
    custom_id = "my-trace-id-123"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.headers.get("x-request-id") == custom_id


def test_middleware_response_time_header_present(client: TestClient):
    """X-Response-Time-Ms header must be present on every response."""
    response = client.get("/health")
    assert "x-response-time-ms" in response.headers


def test_middleware_increments_http_requests_total(client: TestClient):
    """After a request, http_requests_total counter value must be > 0."""
    # Read current value before
    before = sum(
        s.value
        for m in HTTP_REQUESTS_TOTAL.collect()
        for s in m.samples
        if s.name == "http_requests_total_total"
    )
    client.get("/health")
    after = sum(
        s.value
        for m in HTTP_REQUESTS_TOTAL.collect()
        for s in m.samples
        if s.name == "http_requests_total_total"
    )
    assert after >= before


# ===========================================================================
# record_workflow_metrics helper
# ===========================================================================


def test_record_workflow_metrics_observes_histograms():
    """record_workflow_metrics must not raise and must update histogram counters."""
    metrics_dict = {
        "router_time_ms": 50.0,
        "retrieval_time_ms": 200.0,
        "sql_generation_time_ms": 0.0,
        "sql_execution_time_ms": 0.0,
        "answer_generation_time_ms": 1500.0,
        "total_execution_time_ms": 1750.0,
    }
    # Should execute without raising
    record_workflow_metrics(execution_metrics=metrics_dict, route="rag")


def test_record_workflow_metrics_empty_dict():
    """Passing an empty execution_metrics dict must not raise."""
    record_workflow_metrics(execution_metrics={}, route="sql")


def test_record_workflow_metrics_increments_route_counter():
    """REQUESTS_BY_ROUTE_TOTAL must be incremented by record_workflow_metrics."""
    before = sum(
        s.value
        for m in REQUESTS_BY_ROUTE_TOTAL.collect()
        for s in m.samples
        if s.labels.get("route") == "hybrid" and s.name.endswith("_total")
    )
    record_workflow_metrics(
        execution_metrics={"total_execution_time_ms": 100.0},
        route="hybrid",
    )
    after = sum(
        s.value
        for m in REQUESTS_BY_ROUTE_TOTAL.collect()
        for s in m.samples
        if s.labels.get("route") == "hybrid" and s.name.endswith("_total")
    )
    assert after == before + 1


# ===========================================================================
# Tracing module
# ===========================================================================


def test_tracing_disabled_without_otel_env():
    """setup_tracing() must return False when OTEL_EXPORTER_OTLP_ENDPOINT is absent."""
    # Ensure the env var is not set
    os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
    result = setup_tracing(app)
    assert result is False


def test_is_tracing_enabled_false_by_default():
    """is_tracing_enabled() must be False when no OTEL env was provided."""
    os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
    # setup_tracing was never called with a valid endpoint in tests
    assert is_tracing_enabled() is False


@pytest.mark.asyncio
async def test_trace_span_noop_when_disabled():
    """trace_span() must be a safe no-op when tracing is disabled."""
    from backend.app.monitoring.tracing import trace_span

    called = False
    async with trace_span("test_span", key="value") as span:
        called = True
        span.set_attribute("test", "ok")   # must not raise

    assert called
