# MindBridge — Production Environment Variables

This document defines all configuration settings and environment variables recognized by MindBridge in a production hosting environment.

---

## 1. Application & Server Settings

| Variable | Type | Default | Required in Production | Description |
| :--- | :--- | :--- | :--- | :--- |
| `PORT` | integer | `10000` | Yes (Render provides) | Port that Nginx binds to for public HTTP/HTTPS traffic. |
| `ENVIRONMENT` | string | `development` | **Yes (`production`)** | Execution mode. When set to `production`, debug mode is disabled, Swagger docs are disabled, and strict security validation is activated. |
| `DEBUG` | boolean | `true` | **Yes (`false`)** | Disables verbose tracebacks in HTTP error responses. |
| `HOST` | string | `127.0.0.1` | No | Internal bind address for Uvicorn behind Nginx. |
| `API_PREFIX` | string | `/api/v1` | No | Prefix for all backend API endpoints. |
| `LOG_LEVEL` | string | `INFO` | No | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |

---

## 2. Security & Authentication

| Variable | Type | Default | Required in Production | Description |
| :--- | :--- | :--- | :--- | :--- |
| `SECRET_KEY` | string | `""` | **Yes (min 32 chars)** | Secret key used for cryptographic signing of session and auth tokens. |
| `CORS_ORIGINS` | string / JSON list | `["http://localhost:5173"]` | Optional | Allowed CORS origins. Auto-populated with `FRONTEND_BASE_URL` and `RENDER_EXTERNAL_URL`. |
| `FRONTEND_BASE_URL` | string | `http://localhost:5173` | Optional | Public HTTPS origin of the frontend (e.g. `https://mindbridge.onrender.com`). |
| `ACCESS_TOKEN_EXPIRE_MINUTES`| integer | `30` | No | Lifespan of JWT access tokens. |
| `REFRESH_TOKEN_EXPIRE_DAYS` | integer | `7` | No | Lifespan of refresh tokens in HttpOnly cookies. |

---

## 3. Machine Learning & Inference

| Variable | Type | Default | Required in Production | Description |
| :--- | :--- | :--- | :--- | :--- |
| `INFERENCE_DEVICE` | string | `auto` | Optional (`cpu`) | Target execution hardware: `cpu`, `cuda`, or `auto`. On Render, set to `cpu`. |
| `MODEL_LOAD_ON_STARTUP` | boolean | `false` | Optional (`false`) | When `false`, models load lazily upon first inference request, ensuring rapid container startup and passing initial health checks. |
| `INFERENCE_BATCH_SIZE` | integer | `8` | No | Maximum batch size for inference requests. |
| `MAX_REQUEST_TEXT_LENGTH` | integer | `20000` | No | Maximum character length accepted per chat message. |
| `REQUEST_TIMEOUT_SECONDS` | integer | `60` | No | Timeout for synchronous inference execution. |

---

## 4. Privacy & Persistence

| Variable | Type | Default | Required in Production | Description |
| :--- | :--- | :--- | :--- | :--- |
| `STORE_CONVERSATIONS` | boolean | `false` | **Yes (`false`)** | Prevents writing user chat messages to disk, ensuring privacy-by-design for anonymous users. |
| `STORE_ANALYTICS` | boolean | `false` | **Yes (`false`)** | Prevents logging or persisting user-identifiable telemetry. |
| `ENABLE_REQUEST_LOGGING` | boolean | `true` | No | Logs HTTP access metadata (method, path, status, latency) without message body content. |
| `SQLITE_DATABASE_PATH` | path | `backend/data/mental_wellness.db` | No | Path to SQLite database used for credential/auth storage. |

---

## 5. Email & SMTP (Optional)

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `SMTP_HOST` | string | `""` | SMTP mail server hostname (only required if email verification is enabled). |
| `SMTP_PORT` | integer | `587` | SMTP port. |
| `SMTP_USERNAME` | string | `""` | SMTP user login. |
| `SMTP_PASSWORD` | string | `""` | SMTP authentication secret. |
| `SMTP_FROM_EMAIL` | string | `""` | Sender email address. |
| `SMTP_STARTTLS` | boolean | `true` | Enable STARTTLS security. |
