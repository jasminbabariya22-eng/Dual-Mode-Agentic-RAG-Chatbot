from fastapi import FastAPI, Request, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import make_asgi_app
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.app.config import settings
from backend.app.core.logger import logger
from backend.app.api.chat import router as chat_router
from backend.app.monitoring.health import router as health_router
from backend.app.monitoring.middleware import ObservabilityMiddleware
from backend.app.monitoring.tracing import setup_tracing

# Application factory

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "A production-grade Dual-Mode Agentic RAG Chatbot that intelligently "
        "routes user questions to either a Vector RAG pipeline, a Text-to-SQL "
        "engine, or a hybrid combination of both."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json" if settings.APP_ENV != "production" or settings.DEBUG else None,
)

# Rate Limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware  (must be added before observability so CORS headers appear
# even on requests that fail at the middleware layer)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"] if settings.APP_ENV != "production" else ["yourdomain.com", "*.yourdomain.com", "localhost", "127.0.0.1"]
)

# Security Headers & Size Limit Middleware

class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Enforce max request size (e.g., 5MB)
        if request.headers.get("content-length"):
            if int(request.headers["content-length"]) > 5_242_880:
                return JSONResponse(status_code=413, content={"error": "Payload too large"})
                
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        if settings.APP_ENV == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

app.add_middleware(SecurityMiddleware)

# Observability middleware  (wraps every request for metrics + logging)

app.add_middleware(ObservabilityMiddleware)

# Prometheus metrics endpoint  (mounted as a sub-application at /metrics)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Routers

app.include_router(chat_router, prefix="/api/v1")
app.include_router(health_router)   # prefix="/health" is set inside the router

# Optional OpenTelemetry tracing

setup_tracing(app)

# Global exception handlers

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("[API] Unhandled exception: %s", exc)
    
    # Hide details in production
    detail = str(exc) if settings.APP_ENV != "production" or settings.DEBUG else "Internal server error"
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": detail},
    )

# Uvicorn entry point (python -m backend.main)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
