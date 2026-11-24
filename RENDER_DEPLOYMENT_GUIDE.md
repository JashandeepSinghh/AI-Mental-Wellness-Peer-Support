# MindBridge — Render Public Production Deployment Guide

This guide describes the complete procedure for deploying MindBridge (AI Mental Wellness Peer Support) as a public, production-grade web application on [Render](https://render.com).

---

## 1. Architecture Overview

MindBridge is packaged as a **Unified Docker Web Service** containing:
1. **Frontend**: React 18 + TypeScript + Vite SPA, pre-compiled and served directly via Nginx.
2. **Reverse Proxy**: Nginx listening on Render's dynamic `$PORT` (default `10000`), routing static asset requests to `/var/www/html` and API requests `/api/*` to the local backend.
3. **Backend**: FastAPI running on `127.0.0.1:8000` via Uvicorn.
4. **Machine Learning Pipeline**: All 8 integrated ML analysis modules executing real inference on CPU:
   - GoEmotions DistilBERT (28 emotion labels)
   - TweetEval DistilBERT (3 sentiment classes)
   - TweetEval BiLSTM (custom PyTorch recurrent model)
   - Dreaddit DistilBERT (stress detection)
   - FIG DistilBERT (loneliness detection)
   - Context-aware Risk Engine (crisis & safety detection)
   - C-CBT Rumination Detection (cognitive cycle analysis)
   - NRC Valence-Arousal-Dominance (lexicon-based affective metrics)

---

## 2. Resource & Compute Requirements

| Component | Resource Requirement | Justification |
| :--- | :--- | :--- |
| **Active ML Models RAM** | ~2.02 GB checkpoints, ~2.25 GB in-memory | 5 PyTorch/DistilBERT transformer models loaded in memory |
| **FastAPI / Uvicorn / OS** | ~350 MB | Python runtime, Nginx worker processes, buffer memory |
| **Total Minimum RAM** | **~2.8 GB** | Peak allocation during concurrent inference |
| **Recommended Render Plan** | **Standard** (2 CPU, 4 GB RAM) | The Free (512 MB) and Starter (512 MB) tiers will fail with OOM crashes. |
| **Estimated Cost** | ~$25 / month | On-demand Web Service with Git LFS support |

---

## 3. Pre-Deployment Setup: Git LFS & Repository

The model weights total ~2.01 GB and are tracked using **Git Large File Storage (Git LFS)**.

### Git LFS Tracked Patterns (`.gitattributes`)
```
models/**/*.safetensors filter=lfs diff=lfs merge=lfs -text
models/**/*.pt filter=lfs diff=lfs merge=lfs -text
models/**/*.bin filter=lfs diff=lfs merge=lfs -text
```

### Steps to Push to GitHub:
```bash
# 1. Ensure Git LFS is installed and initialized
git lfs install

# 2. Add remote repository (create repository on github.com first)
git remote add origin https://github.com/<your-username>/MindBridge.git

# 3. Stage, commit, and push
git add .
git commit -m "feat: complete MindBridge production release for Render"
git branch -M main
git push -u origin main
```
*Note: Ensure your GitHub account has Git LFS bandwidth/storage enabled.*

---

## 4. Deploying on Render

### Option A: Using Render Blueprint (`render.yaml`) — Recommended

1. Log into your Render Dashboard at [dashboard.render.com](https://dashboard.render.com).
2. Click **Blueprints** in the top navigation.
3. Click **New Blueprint Instance**.
4. Connect your GitHub repository (`MindBridge`).
5. Render will automatically parse `render.yaml` and configure:
   - Service Type: **Web Service (Docker)**
   - Name: `mindbridge`
   - Plan: `standard` (4 GB RAM)
   - Health Check Path: `/api/v1/health`
   - Dynamic Port: `10000`
6. Click **Apply**. Render will clone the repository, download the LFS model weights, build the multi-stage Docker image, and launch the service.

### Option B: Manual Web Service Creation

1. In Render Dashboard, click **New +** > **Web Service**.
2. Select your `MindBridge` GitHub repository.
3. Set the following fields:
   - **Name**: `mindbridge`
   - **Region**: `Oregon (US West)` (or closest to your users)
   - **Branch**: `main`
   - **Root Directory**: Leave blank (root of repo)
   - **Runtime**: `Docker`
   - **Dockerfile Path**: `./Dockerfile`
   - **Instance Type**: `Standard` (2 CPU, 4 GB RAM)
4. Add the following **Environment Variables**:
   | Key | Value | Notes |
   | :--- | :--- | :--- |
   | `PORT` | `10000` | Render defaults to dynamic port |
   | `ENVIRONMENT` | `production` | Enforces production security |
   | `DEBUG` | `false` | Disables debug stack traces & OpenAPI docs |
   | `SECRET_KEY` | *(Generate a 32+ char random string)* | Used for cryptographic tokens |
   | `INFERENCE_DEVICE` | `cpu` | Disables CUDA overhead |
   | `MODEL_LOAD_ON_STARTUP` | `false` | Lazy loads models per request to prevent startup timeouts |
   | `STORE_CONVERSATIONS` | `false` | Privacy-first anonymous mode |
   | `STORE_ANALYTICS` | `false` | Disables telemetry storage |
   | `LOG_LEVEL` | `INFO` | Sanitized production logging |
5. Set **Health Check Path** to `/api/v1/health`.
6. Click **Create Web Service**.

---

## 5. Deployment Verification

Once deployed, Render provides a public HTTPS URL (e.g. `https://mindbridge.onrender.com`).

1. **Health Check Probe**:
   ```bash
   curl -I https://<your-service>.onrender.com/api/v1/health
   # Expected: HTTP/1.1 200 OK
   ```

2. **Public Web Chat Verification**:
   - Open `https://<your-service>.onrender.com/` in an incognito browser window.
   - The chatbot interface appears immediately without requiring login or registration.
   - Send: `"I am feeling stressed about upcoming examinations."`
   - Verify the supportive response and inspect the AI Insights panel to see real inference metrics from the 8 modules.
   - Send crisis test: `"I want to end my life."`
   - Verify immediate crisis escalation with Tele-MANAS (14416), KIRAN (1800-599-0019), and 988 Lifeline.
