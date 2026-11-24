# MindBridge AI Mental Wellness Backend Dockerfile
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies (CPU-optimized PyTorch by default for container portability)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /app/backend/requirements.txt

# Copy application code, scripts, configuration, and model checkpoints
COPY backend/ /app/backend/
COPY scripts/ /app/scripts/
COPY config/ /app/config/
COPY models/ /app/models/
COPY training/ /app/training/
COPY datasets/processed/ /app/datasets/processed/
COPY alembic.ini /app/alembic.ini

# Create directory for persistent SQLite data
RUN mkdir -p /app/backend/data && chmod 777 /app/backend/data

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=5 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

# Start FastAPI application
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
