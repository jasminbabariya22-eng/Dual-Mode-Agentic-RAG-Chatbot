#!/bin/bash
# stop_docker.sh - Stop and remove docker containers

echo "Stopping Docker Compose services..."
docker compose down

echo "To also remove volumes (wipes database and logs), run: docker compose down -v"
