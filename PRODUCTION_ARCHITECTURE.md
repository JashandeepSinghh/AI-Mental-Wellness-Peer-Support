# MindBridge — Production Architecture

This document describes the production system architecture, networking topology, request lifecycles, and component interactions for MindBridge deployed on Render.

---

## 1. High-Level Architecture Diagram

```
                              PUBLIC INTERNET
                                     │
                                     ▼ (HTTPS: Port 443)
                    ┌──────────────────────────────────┐
                    │       Render Cloud Edge          │
                    │   (SSL Termination / DDoS / CDN) │
                    └──────────────────────────────────┘
                                     │
                                     ▼ (HTTP: $PORT e.g. 10000)
┌────────────────────────────────────────────────────────────────────────┐
│                        MINDBRIDGE DOCKER CONTAINER                     │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                          Nginx Proxy                           │   │
│   │              (Listens on $PORT e.g. 10000)                     │   │
│   └───────────────┬────────────────────────────────┬───────────────┘   │
│                   │                                │                   │
│                   ▼                                ▼                   │
│         Static Assets (`/`)                API Proxy (`/api/*`)        │
│                   │                                │                   │
│                   ▼                                ▼                   │
│   ┌────────────────────────────────┐   ┌───────────────────────────┐   │
│   │      React 18 SPA Files        │   │    FastAPI Application    │   │
│   │      (/var/www/html)           │   │    (127.0.0.1:8000)       │   │
│   │ - index.html                   │   └─────────────┬─────────────┘   │
│   │ - bundle.js (Vite + Tailwind)  │                 │                 │
│   │ - SPA fallback routing         │                 ▼                 │
│   └────────────────────────────────┘   ┌───────────────────────────┐   │
│                                        │     InferencePipeline     │   │
│                                        │      (ModelManager)       │   │
│                                        └─────────────┬─────────────┘   │
│                                                      │                 │
│         ┌──────────────────┬──────────────────┬──────┴───────────┐     │
│         ▼                  ▼                  ▼                  ▼     │
│   GoEmotions          TweetEval           TweetEval           Dreaddit │
│   DistilBERT          DistilBERT           BiLSTM            DistilBERT│
│   (Emotion)          (Sentiment)         (Sentiment)          (Stress) │
│                                                                        │
│         ┌──────────────────┬──────────────────┬──────────────────┐     │
│         ▼                  ▼                  ▼                  ▼     │
│      FIG DistilBERT   Risk Engine     C-CBT Rumination       NRC VAD   │
│       (Loneliness)    (Crisis/Safety)   (Cycle Engine)      (Affective)│
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Key Architectural Decisions

### 1. Unified Single-Container Strategy
- **Why**: Eliminates CORS complexity, multiple subdomain management, and API version mismatches.
- **How**: Nginx serves compiled static files directly from local storage with aggressive gzip compression and proxies `/api/` requests internally to `127.0.0.1:8000` with streaming and upgrade headers.
- **Advantage**: Single billable service on Render while retaining strict separation of concerns internally.

### 2. Anonymous-First Interaction Model
- **Public Entry Point**: The home route (`/`) and chat route (`/chat`) are public without any auth wall.
- **Client Session Identification**: The frontend assigns an RFC-compliant UUID `conversation_id` upon initialization using browser `crypto.randomUUID()`.
- **Zero Raw Conversation Persistence**: For public users, chat messages are evaluated in-memory and responses are generated without writing transcripts to disk, guaranteeing privacy-by-design.

### 3. Machine Learning Inference Pipeline
- **Hardware Isolation**: Optimized for CPU inference via PyTorch CPU wheels, avoiding large CUDA shared libraries.
- **Lazy Initialization**: Models are initialized on demand with thread-safe locks (`threading.Lock()`), allowing fast container boot times (<5 seconds) so that initial health checks pass immediately.
- **Memory Management**: Checkpoints are loaded once into shared memory and cached across incoming requests.

---

## 3. Network & Security Topology

| Zone | Protocol | Address | Description |
| :--- | :--- | :--- | :--- |
| **External** | HTTPS (TLS 1.3) | `https://*.onrender.com` | Handled by Render Edge with automatic Let's Encrypt certificates. |
| **Container Ingress**| HTTP | `0.0.0.0:$PORT` | Managed by Nginx. Render forwards public traffic here. |
| **Internal Ingress** | HTTP | `127.0.0.1:8000` | Loopback communication between Nginx and Uvicorn. Not exposed externally. |

---

## 4. Safety & Crisis Escalation Architecture

When any user message is processed:
1. The **Risk Detection Engine** performs context-aware lexical and heuristic evaluation.
2. If crisis indicators (suicidal ideation, severe self-harm, immediate danger) are detected:
   - The response status is escalated to `crisis`.
   - The `SupportiveResponseService` synthesizes an immediate empathetic safety intervention.
   - Verified hotlines are injected into the response:
     - **India**: Tele-MANAS (`14416` / `1800-891-4416`), KIRAN (`1800-599-0019`), Vandrevala Foundation (`+91 9999 666 555`)
     - **United States**: 988 Suicide & Crisis Lifeline (`988`), Crisis Text Line (`HOME` to `741741`)
     - **International**: Find A Helpline directory (`findahelpline.com`)
   - Clinical disclaimers are preserved.
