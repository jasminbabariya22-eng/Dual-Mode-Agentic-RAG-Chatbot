#!/bin/bash
# run.sh - Local development startup script
set -e

echo "Starting local environment..."

if [ ! -f .env ]; then
    echo "Creating .env from .env.example"
    cp .env.example .env
fi

# Load variables
source .env

echo "Running checks..."
black --check backend/
isort --check-only backend/
flake8 backend/
mypy backend/app
pytest backend/tests/

echo "Checks passed. Starting uvicorn..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
