#!/bin/bash
set -e

# Default to 10000 if PORT is not set by Render
PORT="${PORT:-10000}"
export PORT

echo "======================================================"
echo "MindBridge AI Mental Wellness - Production Startup"
echo "Listening Port: ${PORT}"
echo "======================================================"

# 1. Substitute PORT into Nginx configuration
mkdir -p /etc/nginx/conf.d
envsubst '${PORT}' < /etc/nginx/templates/default.conf.template > /etc/nginx/conf.d/default.conf

# 2. Start FastAPI backend via Uvicorn in the background on 127.0.0.1:8000
echo "Starting Uvicorn backend process..."
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# 3. Wait for backend to be healthy before opening Nginx traffic
echo "Waiting for backend health check at http://127.0.0.1:8000/api/v1/health..."
MAX_RETRIES=60
RETRY_COUNT=0
HEALTHY=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -f http://127.0.0.1:8000/api/v1/health > /dev/null 2>&1; then
        echo "Backend health check succeeded after ${RETRY_COUNT}s."
        HEALTHY=1
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT+1))
    sleep 1
done

if [ $HEALTHY -eq 0 ]; then
    echo "ERROR: Backend failed to become healthy within ${MAX_RETRIES} seconds."
    kill -9 $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# 4. Start Nginx in background
echo "Starting Nginx reverse proxy on port ${PORT}..."
nginx -g "daemon off;" &
NGINX_PID=$!

# 5. Handle shutdown signals gracefully
cleanup() {
    echo "Caught termination signal! Shutting down gracefully..."
    kill -TERM $NGINX_PID 2>/dev/null || true
    kill -TERM $BACKEND_PID 2>/dev/null || true
    wait $NGINX_PID 2>/dev/null || true
    wait $BACKEND_PID 2>/dev/null || true
    echo "MindBridge processes terminated."
    exit 0
}

trap cleanup SIGTERM SIGINT SIGQUIT

# 6. Wait for any service process to terminate
wait -n $BACKEND_PID $NGINX_PID
EXIT_STATUS=$?
echo "A service process exited with status ${EXIT_STATUS}. Initiating shutdown..."
cleanup
