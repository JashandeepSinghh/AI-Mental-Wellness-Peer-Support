# ============================================================
# Stage 1: Build Frontend SPA
# ============================================================
FROM node:22-alpine AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci || npm install

COPY frontend/ ./
RUN npm run build

# ============================================================
# Stage 2: Production Unified Image (FastAPI + Nginx)
# ============================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=10000

WORKDIR /app

# Install system dependencies, Nginx, gettext-base (for envsubst), curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    nginx \
    gettext-base \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies with CPU-optimized PyTorch
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /app/backend/requirements.txt

# Copy compiled frontend from Stage 1 into Nginx web root
COPY --from=frontend-builder /app/frontend/dist /var/www/html

# Copy Nginx template and startup script
COPY docker/nginx.conf.template /etc/nginx/templates/default.conf.template
COPY docker/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Copy backend application, models, scripts, config, datasets, alembic
COPY backend/ /app/backend/
COPY scripts/ /app/scripts/
COPY config/ /app/config/
COPY models/ /app/models/
COPY training/ /app/training/
COPY datasets/processed/ /app/datasets/processed/
COPY alembic.ini /app/alembic.ini

# Ensure directories exist and permissions are set
RUN mkdir -p /app/backend/data && chmod -R 777 /app/backend/data
RUN mkdir -p /run/nginx /var/log/nginx /var/cache/nginx

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT:-10000}/api/v1/health || exit 1

CMD ["/app/start.sh"]
