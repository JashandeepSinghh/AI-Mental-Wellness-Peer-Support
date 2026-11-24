# AI Mental Wellness and Peer Support Platform

## Overview

This repository implements a full‑stack AI‑powered mental wellness platform.

- **Backend**: FastAPI, Python, SQLite, authentication, consent management, NLP pipelines, and a supportive‑response service.
- **Frontend**: React + TypeScript SPA built with Vite, offering a calm, professional UI for registration, login, chat, resources, analytics, and profile management.
- **Docker**: Multi‑service Docker Compose setup for easy local development and production deployment.
- **Tests**: Backend unit/integration tests (pytest) and frontend Jest tests ensure core functionality.
- **CI**: GitHub Actions workflow runs lint, tests, builds Docker images, and performs a health‑check.

## Quick Start (Local Development)

### Prerequisites
- **Node.js** (v20+ recommended)
- **Python** (3.10+)
- **Docker Desktop** with Compose plugin (or `docker-compose`)

### Backend
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
The API will be available at `http://localhost:8000/api/v1/`.

### Frontend
```bash
cd frontend
npm ci
npm run dev   # Vite dev server on http://localhost:5173
```
The development proxy forwards `/api/` calls to the backend.

### Run All Services with Docker
```bash
docker compose build   # builds backend and frontend images
docker compose up -d   # starts containers
# Health checks
curl http://localhost:8000/api/v1/health
curl http://localhost/   # returns the SPA index
```
Stop the stack with `docker compose down`.

## CI / GitHub Actions
The workflow defined in `.github/workflows/ci.yml` runs on every push:
- Installs dependencies
- Executes backend tests (`pytest`)
- Executes frontend tests (`npm test`)
- Builds Docker images
- Starts the compose stack and checks health endpoints

## Project Structure
```
/backend
    app/                # FastAPI application
    supportive_response_service.py
    requirements.txt
    Dockerfile
/frontend
    src/                # React source
    public/
    package.json
    Dockerfile
    nginx.conf
docker-compose.yml
README.md
PROJECT_PROGRESS.md
```

## API Contract (excerpt)
`POST /api/v1/chat`
```json
{
  "response_text": "...",
  "detected_emotions": ["sadness", "anxiety"],
  "stress_detected": true,
  "loneliness_detected": false,
  "wellness_resources": ["Mindful breathing exercise", "Grounding techniques"],
  "safety_guidance": {
    "isUrgent": true,
    "message": "If you are in immediate danger, call 988...",
    "crisis_contacts": [{"name":"National Suicide Prevention Lifeline","phone":"988"}]
  }
}
```

## License
MIT License.
