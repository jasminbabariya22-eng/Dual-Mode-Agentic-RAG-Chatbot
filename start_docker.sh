#!/bin/bash
# start_docker.sh - Production docker startup

set -e

if [ ! -f .env ]; then
    echo "Warning: .env file missing. Using .env.example as template."
    cp .env.example .env
fi

echo "Verifying Docker..."
if ! command -v docker &> /dev/null; then
    echo "Docker could not be found. Please install Docker."
    exit 1
fi

echo "Starting Docker Compose services in detached mode..."
docker compose up -d --build

echo "Waiting for API healthcheck to pass..."
sleep 5

MAX_RETRIES=10
RETRY_COUNT=0
until curl -s http://localhost:8000/health/live > /dev/null || [ $RETRY_COUNT -eq $MAX_RETRIES ]; do
    echo "Waiting for API... ($RETRY_COUNT / $MAX_RETRIES)"
    sleep 3
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "API failed to become healthy. Check logs:"
    docker compose logs api
    exit 1
fi

echo "All services successfully started and API is healthy!"
echo "API is running on port 8000 (and nginx on port 80)."
echo "Grafana is running on port 3000."
