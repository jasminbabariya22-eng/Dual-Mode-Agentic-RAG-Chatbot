"""
Health & Readiness Probe Endpoints.

Three endpoints surfaced under the /health prefix:

    GET /health        — Liveness. Always returns 200 {"status":"healthy"}.
    GET /health/live   — Alias for liveness.
    GET /health/ready  — Readiness. Checks all downstream dependencies.

Readiness check design:
  - Each dependency is checked independently and in parallel (asyncio.gather).
  - A failing Redis check yields "degraded" (not "failed") because the
    in-memory fallback is always active.
  - A failing LLM check yields "degraded" because the Groq fallback is
    configured.
  - SQLite failure or ChromaDB failure yields "unhealthy" — these are hard
    dependencies with no fallback.
  - Overall status: "healthy" if all ok, "degraded" if any DEGRADED, "unhealthy"
    if any hard failure.  HTTP 200 for healthy/degraded, 503 for unhealthy.
"""

import asyncio
import sqlite3
import time
from typing import Any, Dict, Tuple

import httpx
from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from backend.app.config import settings
from backend.app.core.logger import logger

router = APIRouter(prefix="/health", tags=["Health"])

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class CheckResult(BaseModel):
    status: str          # "ok" | "degraded" | "failed"
    detail: str = ""


class ReadinessResponse(BaseModel):
    status: str          # "healthy" | "degraded" | "unhealthy"
    checks: Dict[str, CheckResult]
    version: str
    checked_at: str


# ---------------------------------------------------------------------------
# Individual dependency checks
# ---------------------------------------------------------------------------


async def _check_sqlite() -> Tuple[str, CheckResult]:
    """Verify the SQLite orders database is reachable and not corrupted."""
    try:
        db_path = settings.SQLITE_DB_PATH
        if not db_path.exists():
            return "sqlite", CheckResult(
                status="failed",
                detail=f"Database file not found: {db_path}",
            )
        # Run in executor to avoid blocking the event loop
        def _probe():
            conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check;")
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else "unknown"

        result = await asyncio.get_event_loop().run_in_executor(None, _probe)
        if result == "ok":
            return "sqlite", CheckResult(status="ok")
        return "sqlite", CheckResult(status="failed", detail=f"integrity_check returned: {result}")
    except Exception as exc:
        return "sqlite", CheckResult(status="failed", detail=str(exc))


async def _check_vector_store() -> Tuple[str, CheckResult]:
    """Verify ChromaDB is reachable by listing collections."""
    try:
        import chromadb

        def _probe():
            client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_PATH))
            collections = client.list_collections()
            return len(collections)

        count = await asyncio.get_event_loop().run_in_executor(None, _probe)
        return "vector_store", CheckResult(
            status="ok",
            detail=f"{count} collection(s) found",
        )
    except Exception as exc:
        return "vector_store", CheckResult(status="failed", detail=str(exc))


async def _check_redis() -> Tuple[str, CheckResult]:
    """
    Ping Redis.  Failure is reported as 'degraded' — not 'failed' — because
    the application has a working in-memory fallback for all cache operations.
    """
    try:
        import redis as redis_lib

        def _probe():
            client = redis_lib.from_url(settings.REDIS_URL, socket_timeout=2.0)
            return client.ping()

        await asyncio.get_event_loop().run_in_executor(None, _probe)
        return "redis", CheckResult(status="ok")
    except Exception as exc:
        return "redis", CheckResult(
            status="degraded",
            detail=f"Redis unreachable ({exc}); in-memory fallback active.",
        )


async def _check_llm() -> Tuple[str, CheckResult]:
    """
    HEAD request to the Ollama base URL.
    Failure → 'degraded' because the Groq fallback LLM is configured.
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(settings.OLLAMA_BASE_URL)
        if resp.status_code < 500:
            return "llm", CheckResult(status="ok", detail=f"Ollama HTTP {resp.status_code}")
        return "llm", CheckResult(
            status="degraded",
            detail=f"Ollama returned HTTP {resp.status_code}; Groq fallback active.",
        )
    except Exception as exc:
        return "llm", CheckResult(
            status="degraded",
            detail=f"Ollama unreachable ({exc}); Groq fallback active.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "",
    summary="Liveness probe",
    description="Always returns HTTP 200 while the process is running.",
)
async def health_liveness() -> Dict[str, str]:
    return {"status": "healthy"}


@router.get(
    "/live",
    summary="Liveness probe (alias)",
    description="Alias of GET /health.",
)
async def health_live() -> Dict[str, str]:
    return {"status": "healthy"}


@router.get(
    "/ready",
    summary="Readiness probe",
    description=(
        "Checks all downstream dependencies. "
        "Returns 200 for healthy/degraded, 503 for unhealthy."
    ),
    response_model=ReadinessResponse,
)
async def health_ready(response: Response) -> ReadinessResponse:
    # Run all checks concurrently
    results = await asyncio.gather(
        _check_sqlite(),
        _check_vector_store(),
        _check_redis(),
        _check_llm(),
    )

    checks: Dict[str, CheckResult] = {name: result for name, result in results}

    # Determine aggregate status
    statuses = {r.status for r in checks.values()}
    if "failed" in statuses:
        overall = "unhealthy"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif "degraded" in statuses:
        overall = "degraded"
        # HTTP 200 — degraded means the system still works
    else:
        overall = "healthy"

    logger.info("[Health] Readiness check: %s | checks=%s", overall, checks)

    return ReadinessResponse(
        status=overall,
        checks=checks,
        version=settings.APP_VERSION,
        checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

@router.get(
    "/startup",
    summary="Startup probe",
    description="Indicates whether the application has started successfully.",
)
async def health_startup() -> Dict[str, str]:
    # In a more complex app, this could wait for initial syncs or cache warming.
    # For now, it simply confirms the app is up.
    return {"status": "started"}
