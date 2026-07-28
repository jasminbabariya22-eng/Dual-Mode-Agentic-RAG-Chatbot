"""
Optional OpenTelemetry Tracing Support.

Tracing is activated only when OTEL environment variables are present.
If they are absent the entire module is a lightweight no-op — no external
library is required at runtime.

Activation:
    Set either of these environment variables before starting the server:
        OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
        OTEL_SERVICE_NAME=rag-chatbot          (optional, defaults to APP_NAME)

Usage (in nodes / handlers — optional):
    from backend.app.monitoring.tracing import trace_span

    async with trace_span("router_classification") as span:
        span.set_attribute("question.length", len(question))
        result = await router.route(question)

If tracing is disabled, trace_span is a no-op async context manager.

Design decisions:
  - All imports are guarded by try/except so the module loads cleanly even
    when opentelemetry packages are absent.
  - setup_tracing() is idempotent — calling it twice is safe.
  - FastAPIInstrumentor auto-instruments all HTTP spans when OTEL is enabled.
"""

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from backend.app.config import settings
from backend.app.core.logger import logger

# Whether tracing has been successfully initialised
_tracing_enabled: bool = False


def setup_tracing(app: Any) -> bool:
    """
    Configure OpenTelemetry tracing if OTEL env vars are present.

    Args:
        app: The FastAPI application instance.

    Returns:
        True if tracing was enabled, False if env vars are absent or setup fails.
    """
    global _tracing_enabled

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        logger.info(
            "[Tracing] OTEL_EXPORTER_OTLP_ENDPOINT not set — tracing disabled."
        )
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        service_name = os.getenv("OTEL_SERVICE_NAME", settings.APP_NAME)

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        # Auto-instrument all FastAPI HTTP handlers
        FastAPIInstrumentor.instrument_app(app)

        _tracing_enabled = True
        logger.info(
            "[Tracing] OpenTelemetry enabled. Service=%s Endpoint=%s",
            service_name,
            endpoint,
        )
        return True

    except ImportError as exc:
        logger.warning("[Tracing] OpenTelemetry packages not available: %s", exc)
        return False
    except Exception as exc:
        logger.error("[Tracing] Failed to configure tracing: %s", exc)
        return False


def is_tracing_enabled() -> bool:
    """Return True if OpenTelemetry tracing was successfully configured."""
    return _tracing_enabled


@asynccontextmanager
async def trace_span(name: str, **attributes: Any) -> AsyncGenerator[Any, None]:
    """
    Async context manager that creates an OTel span when tracing is enabled,
    or behaves as a no-op otherwise.

    Args:
        name:       Span name (e.g. "router_classification").
        attributes: Key-value pairs set as span attributes.

    Usage:
        async with trace_span("sql_generation", question=q) as span:
            result = await engine.generate(q)
    """
    if not _tracing_enabled:
        yield _NoOpSpan()
        return

    try:
        from opentelemetry import trace

        tracer = trace.get_tracer(settings.APP_NAME)
        with tracer.start_as_current_span(name) as span:
            for key, val in attributes.items():
                try:
                    span.set_attribute(key, str(val))
                except Exception:
                    pass
            yield span
    except Exception:
        yield _NoOpSpan()


class _NoOpSpan:
    """Minimal no-op span object returned when tracing is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: D102
        pass

    def record_exception(self, exc: Exception) -> None:  # noqa: D102
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:  # noqa: D102
        pass
