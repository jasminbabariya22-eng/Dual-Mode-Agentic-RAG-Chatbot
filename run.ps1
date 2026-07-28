# run.ps1 - Local development startup script
$ErrorActionPreference = "Stop"

Write-Host "Starting local environment..."

if (!(Test-Path -Path ".env")) {
    Write-Host "Creating .env from .env.example"
    Copy-Item .env.example .env
}

Write-Host "Running checks..."
black --check backend/
if ($LASTEXITCODE -ne 0) { throw "Black failed" }
isort --check-only backend/
if ($LASTEXITCODE -ne 0) { throw "Isort failed" }
flake8 backend/
if ($LASTEXITCODE -ne 0) { throw "Flake8 failed" }
mypy backend/app
if ($LASTEXITCODE -ne 0) { throw "Mypy failed" }
pytest backend/tests/
if ($LASTEXITCODE -ne 0) { throw "Pytest failed" }

Write-Host "Checks passed. Starting uvicorn..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
