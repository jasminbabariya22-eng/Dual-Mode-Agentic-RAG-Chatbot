# Stage 1: Builder
FROM python:3.11-slim AS builder

# Set environment variables for the build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies into a local directory
COPY backend/requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 2: Final Production Image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    APP_ENV=production

# Create a non-root user
RUN adduser --disabled-password --gecos '' appuser

WORKDIR /app

# Install runtime dependencies only (if any are needed outside of python)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed python dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy the application code
COPY backend/ /app/backend/

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Healthcheck to ensure FastAPI is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health/live || exit 1

# Expose port
EXPOSE 8000

# Start Uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
