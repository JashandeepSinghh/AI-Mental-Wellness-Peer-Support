# Stage 1: Build the React + TypeScript frontend
FROM node:22-alpine AS builder

WORKDIR /app

# Install dependencies
COPY frontend/package*.json ./
RUN npm ci || npm install

# Build the SPA bundle
COPY frontend/ ./
RUN npm run build

# Stage 2: Serve via Nginx with reverse proxy
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q -O - http://127.0.0.1/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
