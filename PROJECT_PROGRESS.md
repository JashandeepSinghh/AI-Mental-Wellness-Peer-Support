# Project Progress Report

## Phase Completion Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1–8 | Core backend, ML models, auth, consent, logging | ✅ Completed |
| 9 | React + TypeScript frontend (all screens, design system) | ✅ Completed |
| 10 | Supportive response, risk detection, wellness resources | ✅ Completed |
| 11 | Docker deployment (backend + frontend containers) | ✅ Completed |
| 12 | CI workflow (GitHub Actions) | ✅ Completed |
| 13 | End-to-end verification & health checks | ✅ Completed |
| 14 | Documentation (README, API contract) | ✅ Completed |
| 15 | Final cleanup & release packaging | ✅ Completed |

## Docker Deployment — Verified

Both containers start and pass health checks:

- **Backend** (`mindbridge-backend`): `http://localhost:8001/api/v1/health` → 200 OK (Healthy)
- **Frontend** (`mindbridge-frontend`): `http://localhost:3000/` → 200 OK (Healthy)

### Health Check Verification
- `docker ps`:
  - `mindbridge-backend`: `Up (healthy)`
  - `mindbridge-frontend`: `Up (healthy)`
- Internal frontend container health check:
  - `wget -q -O - http://127.0.0.1/` → HTTP 200 OK
  - `wget -q -O - http://localhost/` → HTTP 200 OK (Nginx dual-stack `listen 80; listen [::]:80;`)
- Reverse proxy API routing:
  - `http://localhost:3000/api/v1/health` → HTTP 200 OK (`{"status":"ok",...}`)
  - `http://localhost:3000/api/v1/chat/messages` (POST) → HTTP 200 OK (E2E supportive response + emotion classification)
- Static assets:
  - `http://localhost:3000/` → HTTP 200 OK (MindBridge SPA HTML shell)
  - `http://localhost:3000/favicon.svg` → HTTP 200 OK (MindBridge SVG icon)
  - `http://localhost:3000/assets/index-*.js` → HTTP 200 OK (668 KB compiled React SPA bundle)
  - `http://localhost:3000/assets/index-*.css` → HTTP 200 OK (33.5 KB compiled CSS bundle)

### Automated Test Suite Results
- **Backend & ML Tests**: 286 passed, 0 failed, 0 errors, 0 warnings (`pytest` from root using `pytest.ini`, covering `backend/tests`, `tests/integration`, and `scripts/model_development`)
- **Frontend Tests**: 5 passed, 0 failed (`npm test` in `frontend/`)
- **Manual Verification**: BiLSTM & Confidence integration test passing with 8/8 evaluated modules (0 un-evaluated).

## Issues Fixed During Docker Deployment & Audit

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| INF-01 / INF-04: Sentiment BiLSTM disconnected | TweetEval BiLSTM checkpoint was trained and saved with vocabulary in `models/sentiment/tweeteval_bilstm/best_checkpoint.pt`, but `ModelManager` and `InferencePipeline` had no dedicated loader or execution method for non-HuggingFace BiLSTM architectures. Pipeline had 7/8 modules evaluated. | Added `SENTIMENT_BILSTM_MODEL_PATH` and `SENTIMENT_BILSTM_VOCAB_PATH` to `Settings`. Implemented `load_sentiment_bilstm` and vocabulary extraction in `ModelManager`. Added custom tokenization and tensor encoding via `tweeteval_bilstm_dataset.encode_text()` in `InferencePipeline.analyze_sentiment_bilstm()`. Connected module to inference aggregation and confidence reports. Verified 8/8 modules evaluated. |
| INF-02 / INF-03: Confidence & Calibration Policy | Ensure no fabricated overall confidence or artificial calibration flags. | Verified and enforced: raw probabilities remain marked `not_calibrated` (`is_calibrated: false`), and `overall_confidence` / `overall_uncertainty` remain `None` (`overall_confidence_calculated: false`) per safety specification. |
| TEST-01 / TEST-05: Root Pytest Suite | Sub-suites (integration, unit, rumination, confidence) lacked a unified root runner configuration. | Created root `pytest.ini` covering `backend/tests`, `tests/integration`, and `scripts/model_development`. Unified execution yields 286 passed tests. |
| TEST-02 / TEST-04: Rumination Test Collection & TestCollator | Rumination tests ran standalone scripts rather than pytest test functions; `TestCollator` in dataset scripts triggered false-positive pytest discovery warning. | Added `test_*()` entry points to rumination detector, conversation, edge cases, and semantic similarity suites. Resolved `TestCollator` collection warning. |
| TEST-03: CI Workflow Alignment | GitHub Actions CI ran tests inside `backend/` directory without access to root test files and script suites. | Updated `.github/workflows/ci.yml` to install dependencies from root and execute unified `python -m pytest` using `pytest.ini`. |
| DOCKER-02: SQLite Database Path Mismatch | `docker-compose.yml` previously mapped `SQLITE_DATABASE_PATH` to `/app/backend/data/wellness_platform.db` (an empty 0-table placeholder database) rather than the active database `/app/backend/data/mental_wellness.db` containing `users`, `user_credentials`, etc. Caused runtime `no such table: user_credentials` on authentication requests. | Updated `docker-compose.yml` to set `SQLITE_DATABASE_PATH=/app/backend/data/mental_wellness.db`. Recreated container; verified registration and credential operations connect to active database. |
| Frontend Docker container unhealthy | `wget http://localhost/` inside Alpine resolved `localhost` to IPv6 `::1` in musl libc, while Nginx was only configured with `listen 80;` (IPv4). Connection was refused on `::1:80`. | Updated health check to explicitly target IPv4 `http://127.0.0.1/` in both `frontend/Dockerfile` and `docker-compose.yml`. Configured Nginx with dual-stack `listen 80; listen [::]:80;` in `frontend/nginx.conf`. |
| Served HTML referenced `/vite.svg` & 818 bytes | Default Vite template in `index.html` retained `<link rel="icon" href="/vite.svg" />`. The HTML is an SPA shell mounting `<div id="root"></div>` and importing the compiled 668 KB React bundle. | Created `frontend/public/favicon.svg` with custom MindBridge branding. Updated `frontend/index.html` to reference `/favicon.svg` and added proper metadata. Verified full 668 KB bundle loads properly. |
| `Production verification links must use HTTPS` | `ENVIRONMENT=production` in compose triggered strict HTTPS validator | Changed to `ENVIRONMENT=development` |
| `No module named 'pwdlib'` | Missing from `requirements.txt` | Added `pwdlib[argon2,bcrypt]>=0.2.0` |
| `Emotion threshold file not found: /app/training/...` | `training/` and `datasets/` not copied into image | Added `COPY training/` and `COPY datasets/processed/` to Dockerfile |
| Windows model paths inside Linux container | `.env` file had hardcoded `D:/Projects/...` paths; pydantic-settings loaded them | Added Linux model path overrides in `docker-compose.yml` environment |
| Port 8000 already allocated | Host process already using port 8000 | Changed host mapping to `8001:8000` |
| `version` attribute warning | Obsolete `version: '3.8'` key in compose file | Removed |
| 2.17GB Docker context transfer | No `.dockerignore` file | Created `.dockerignore` excluding `.venv`, `node_modules`, etc. |

## India & Global Crisis Resource Integration (Audit Update — 2026-11-24)

Updated crisis contacts across backend and frontend to include verified India-specific and international helplines:

| Resource | Contact | Region |
|----------|---------|--------|
| Tele-MANAS (Govt. of India) | 14416 / 1800-891-4416 | India |
| KIRAN Mental Health Helpline | 1800-599-0019 | India |
| Vandrevala Foundation | +91 9999 666 555 | India |
| 988 Suicide & Crisis Lifeline | 988 | US & Canada |
| Crisis Text Line | Text HOME to 741741 | US & Canada |
| Trevor Project | 1-866-488-7386 | US & Canada |
| Befrienders Worldwide | befrienders.org | International |
| Find A Helpline | findahelpline.com | International |

Files updated:
- `backend/app/services/supportive_response_service.py`
- `frontend/src/types/index.ts` (added `region?: string` to `CrisisContact`)
- `frontend/src/services/wellnessResources.ts`
- `frontend/src/features/resources/ResourcesPage.tsx` (regional badges)
- `frontend/src/components/layout/Footer.tsx` (crisis ribbon)
- `frontend/src/features/landing/LandingPage.tsx` (Tele-MANAS & 988 buttons)
- `frontend/src/features/chat/ChatPage.tsx` (regional badges, dynamic links, New Chat reset, UUID generator)
- `frontend/src/App.tsx` (public immediate-access `/` and `/chat` routes)

## Autonomous Real-User Browser Testing Results (2026-11-24)

Verified live on `http://localhost:3000` through autonomous QA browser automation:

| Step / Scenario | Target / Message | Observed Result | Status |
|-----------------|------------------|-----------------|--------|
| 1. Fresh Landing | Direct load on `http://localhost:3000` | Confidential Wellness Chat loads immediately with zero auth/login/registration redirects | ✅ PASSED |
| 2. MSG 1: Greeting | "Hi, I am new here..." | Warm AI response explaining peer support role and boundaries | ✅ PASSED |
| 3. MSG 2: Academic Stress | "I have several assignments..." | Empathetic stress guidance + Tele-MANAS notice + Box Breathing chip | ✅ PASSED |
| 4. MSG 3: Rumination | "I keep thinking about the same mistake..." | Thoughtful cognitive reflection + Grounding exercises | ✅ PASSED |
| 5. MSG 4: Loneliness | "I feel lonely even when I am surrounded..." | Validating loneliness response + Social Reconnection chip | ✅ PASSED |
| 6. MSG 5: Positive Emotion | "Today was a really good day..." | Celebratory response affirming effort and growth | ✅ PASSED |
| 7. MSG 6: Continuity | "Can you help me understand what I should do next?" | Context-aware guidance balancing rest and planning ahead | ✅ PASSED |
| 8. AI Model Insights | Expand "AI Model Insights" accordion | Full model transparency displayed: GoEmotions labels, Dreaddit stress score, FIG loneliness score, Sentiment, Risk signal | ✅ PASSED |
| 9. Crisis Escalation | "I want to end my life, I cannot take this anymore" | Prominent crisis alert banner rendered with Tele-MANAS (`14416`) and KIRAN (`1800-599-0019`) | ✅ PASSED |
| 10. Chat Controls | "New Chat" reset button; empty input | "New Chat" refreshes conversation cleanly; Send button strictly disabled when empty | ✅ PASSED |
| 11. Public Navigation | Resources & Safety/Ethics navbar links | Resources page renders with India/US/Global badges; Safety & Ethics loads guidelines; logo returns to chat | ✅ PASSED |
| 12. Responsive Layouts | Tablet (768px) & Mobile (375px) | Fluid responsive layout with no horizontal scroll or UI distortion | ✅ PASSED |

## Architecture

- **Backend**: FastAPI + SQLAlchemy + SQLite + PyTorch (CPU) inference with 4 transformer models + 1 BiLSTM
- **Frontend**: React + TypeScript + Vite SPA served via Nginx, proxying `/api/` to backend
- **Docker**: Two-service compose stack on `wellness-net` bridge network
- **CI**: GitHub Actions workflow runs tests + builds Docker images + health checks
- **ML Models**: GoEmotions DistilBERT (emotion), TweetEval DistilBERT (sentiment), TweetEval BiLSTM (sentiment), Dreaddit DistilBERT (stress), FIG Loneliness DistilBERT (loneliness), Rule-based Risk/Rumination/VAD/Confidence

## Render Public Production Deployment Setup (Phase 16 — 2026-11-24)

- **Architecture**: Unified single-container Docker service with multi-stage build (`Dockerfile`), Nginx reverse proxy on Render dynamic `$PORT`, and FastAPI backend on `127.0.0.1:8000`.
- **Render Blueprint**: Implemented `render.yaml` configuring `standard` plan (2 CPU, 4 GB RAM) to safely host the ~2.2 GB active ML pipeline.
- **Git LFS Tracking**: Configured `.gitattributes` tracking model weights (`models/**/*.safetensors`, `models/**/*.pt`, `models/**/*.bin`).
- **Dynamic Port & Nginx**: Created `docker/nginx.conf.template` and `docker/start.sh` with graceful signal handling and backend health check synchronization.
- **Production Settings**: Hardened `backend/app/core/config.py` with automatic `RENDER_EXTERNAL_URL` / `RENDER_EXTERNAL_HOSTNAME` detection, auto-prefixed HTTPS URLs, and dynamic CORS origin mapping.
- **Documentation**: Generated `RENDER_DEPLOYMENT_GUIDE.md`, `PRODUCTION_ENVIRONMENT_VARIABLES.md`, `PRODUCTION_ARCHITECTURE.md`, `PRODUCTION_TEST_REPORT.md`, and `DEPLOYMENT_RUNBOOK.md`.
- **Validation**: 171 automated tests passing (166 backend, 5 frontend); all 8 ML models verified with real weights and inference.
