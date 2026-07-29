import json
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from backend.app.core.logger import logger, request_id_var, session_id_var
from backend.app.monitoring.metrics import (
    ACTIVE_SESSIONS_GAUGE,
    ERRORS_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
    STREAMING_REQUESTS_TOTAL,
)

class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Per-request observability: request IDs, Prometheus metrics, structured logs.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 1. Request ID — prefer client-supplied header for trace propagation
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        
        request_id_var.set(request_id)
        
        # Optionally attempt to extract session ID from request body if it's a POST
        # But we'll just set it to "-" by default for middleware logging
        session_id_var.set("-")

        # 2. Track streaming & active session gauges
        is_stream = "/stream" in request.url.path
        if is_stream:
            STREAMING_REQUESTS_TOTAL.inc()

        ACTIVE_SESSIONS_GAUGE.inc()
        start = time.perf_counter()

        # 3. Call downstream
        try:
            response: Response = await call_next(request)
        except Exception as exc:
            ACTIVE_SESSIONS_GAUGE.dec()
            ERRORS_TOTAL.labels(component="middleware", error_type=type(exc).__name__).inc()
            raise

        # 4. Measure & record
        duration_s = time.perf_counter() - start
        status_code = response.status_code

        # Normalise path to avoid high cardinality (/chat/stream → /chat/stream)
        path = request.url.path

        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            path=path,
            status_code=str(status_code),
        ).inc()

        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=request.method,
            path=path,
        ).observe(duration_s)

        if status_code >= 500:
            ERRORS_TOTAL.labels(
                component="http",
                error_type=f"HTTP_{status_code}",
            ).inc()

        ACTIVE_SESSIONS_GAUGE.dec()

                # 5. Structured JSON log
                log_record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level": "INFO" if status_code < 500 else "ERROR",
            "request_id": request_id,
            "method": request.method,
            "path": path,
            "status_code": status_code,
            "duration_ms": round(duration_s * 1000, 2),
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", ""),
        }
        logger.info("[Request] %s", json.dumps(log_record))

                # 6. Inject response headers
                response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(round(duration_s * 1000, 2))

        return response
