# Dual-Mode Agentic RAG Chatbot - Production Deployment

This document covers the infrastructure, architecture, and instructions for deploying the Agentic RAG Chatbot in a production environment.

## 1. Project Architecture

The application is deployed using Docker containers orchestrated by `docker-compose.yml`. 
- **FastAPI (`api`)**: The core application logic, LangGraph workflow, guardrails, and Chat endpoints. Runs via Uvicorn.
- **Nginx (`nginx`)**: Reverse proxy that handles large request buffering, security headers, and compression.
- **Redis (`redis`)**: Used for session memory, cache, and high-speed metadata tracking.
- **Ollama (`ollama`)**: Locally-hosted LLM inference for privacy-first operations.
- **Prometheus & Grafana**: System and application metric scraping and dashboards.

## 2. Environment Variables

Production deployments require explicit configuration via a `.env` file at the repository root. A template is provided in `.env.example`.

### Critical Variables:
- `APP_ENV=production`: Enables security headers, disables Swagger UI, and obfuscates stack traces.
- `GROQ_API_KEY`: Fallback LLM provider key if Ollama goes offline.
- `REDIS_URL=redis://redis:6379/0`: The internal Docker network address for Redis.
- `OLLAMA_BASE_URL=http://ollama:11434`: The internal Docker network address for Ollama.

## 3. Deployment with Docker

We provide helper scripts to simplify the deployment:

### Start Services
```bash
./start_docker.sh
```
This script ensures the `.env` exists, builds the images, brings up all containers in detached mode, and polls the readiness probe until the API is fully alive.

### Stop Services
```bash
./stop_docker.sh
```
*(To wipe persistent volumes like SQLite, Chroma, and Redis, run `docker compose down -v`)*

## 4. CI/CD Pipeline

The repository includes GitHub Actions for automated quality assurance:
- **`ci.yml`**: Runs `black`, `isort`, `flake8`, `mypy`, and `pytest` on every push/PR to `main`. It enforces strict style guides and requires 100% test success.
- **`docker.yml`**: Automatically builds and pushes a version-tagged Docker image to the GitHub Container Registry (`ghcr.io`) upon pushing a new tag (e.g. `v1.0.0`).

## 5. Health Endpoints & Observability

- **Liveness (`/health/live`)**: Returns HTTP 200 immediately if the API container is running.
- **Readiness (`/health/ready`)**: Verifies connectivity to Redis, SQLite, ChromaDB, and Ollama. If any core dependency is down, it returns HTTP 503.
- **Metrics (`/metrics`)**: Exported in Prometheus format. Includes tracking for `http_requests_total`, `streaming_requests_total`, and latency histograms.

## 6. Scaling Recommendations

For large-scale enterprise deployments:
1. **Kubernetes**: Migrate from `docker-compose` to the provided Kubernetes manifests (in `k8s/`).
2. **External Database**: Move the SQLite backend to PostgreSQL.
3. **Dedicated GPU Nodes**: Separate the `ollama` container to a dedicated node pool with GPU acceleration to reduce inference latency.
4. **Load Balancing**: Deploy multiple `api` replicas behind Nginx or a cloud native ingress controller (e.g., NGINX Ingress, AWS ALB).
