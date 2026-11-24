# MindBridge — Production Deployment Runbook

This operational runbook provides instructions for deployment, monitoring, troubleshooting, scaling, and rolling back MindBridge on Render.

---

## 1. Triggering Deployments

Deployments can be executed automatically or manually:

### Automatic Continuous Deployment
- Render monitors the `main` branch of the connected GitHub repository.
- Every push to `main` triggers a new Docker build and rolling deployment.
- Zero-downtime: Render keeps the previous container running until the new container's health check (`/api/v1/health`) returns HTTP 200.

### Manual Re-Deploy
1. Go to **Render Dashboard** > Select `mindbridge`.
2. Click **Manual Deploy** in the top-right corner.
3. Select **Clear build cache & deploy** (recommended if model checkpoints or dependencies changed).

---

## 2. Health Monitoring & Verification

### Endpoints
- **Liveness & Readiness**: `GET https://<service>.onrender.com/api/v1/health`
  - Expected Response: `{"status":"ok","service":"AI Mental Wellness and Peer Support API","version":"1.0.0","environment":"production"}`
- **Frontend Root**: `GET https://<service>.onrender.com/`
  - Expected Response: HTTP 200 with HTML application shell.

### Render Log Inspection
- In the Render Dashboard, open the **Logs** tab.
- Key log milestones to look for during boot:
  ```
  Starting MindBridge Production Container
  Port: 10000
  Starting Uvicorn backend process...
  Backend is ready!
  Starting Nginx reverse proxy on port 10000...
  ```

---

## 3. Incident Management & Common Failures

### Issue A: Container Out of Memory (OOM Kill)
- **Symptom**: Container crashes shortly after receiving the first chat message; log displays `Killed` or `OOMKilled`.
- **Root Cause**: The Render service plan has less than 4 GB RAM (e.g. Free or Starter tier with 512 MB).
- **Remedy**:
  1. Open Render Dashboard > `mindbridge` > **Settings**.
  2. Under **Instance Type**, upgrade from Free/Starter to **Standard** (4 GB RAM).
  3. Restart the service.

### Issue B: Git LFS Pointer Error During Build
- **Symptom**: Python raises `OSError: Unable to load weights from file` or `safetensors_rust.SafetensorError: file is too small`.
- **Root Cause**: Git repository contains LFS text pointer files instead of actual binary weights.
- **Remedy**:
  1. Verify Git LFS is installed locally: `git lfs install`.
  2. Run `git lfs pull` locally to ensure all weights are real binaries.
  3. Re-push with `git push origin main`.
  4. On Render, ensure Git LFS is enabled under repository settings.

### Issue C: Health Check Timeout During Boot
- **Symptom**: Render reports `Timed out waiting for health check at /api/v1/health`.
- **Root Cause**: `MODEL_LOAD_ON_STARTUP` is set to `true`, causing PyTorch to load all 8 models before answering the health check.
- **Remedy**:
  1. Verify that `MODEL_LOAD_ON_STARTUP=false` in environment variables.
  2. With lazy loading, the health check responds within 2 seconds.

---

## 4. Rollback Procedure

If a bad commit causes production regressions:
1. Go to Render Dashboard > `mindbridge` > **Deploys**.
2. Find the previous successful deploy.
3. Click the three dots menu `...` on that deploy and select **Rollback to this deploy**.
4. Render will immediately restore the previous image without rebuilding.
