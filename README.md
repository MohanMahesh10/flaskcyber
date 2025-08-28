# FlaskCyber – Intrusion Detection + Crypto Demo

A lightweight cybersecurity demo application:
- Flask backend (Render-ready) providing IDS, text/CSV/image analysis, charts data, and crypto key generation APIs
- Static frontend (GitHub Pages-ready) in `docs/` that calls the backend via `?api=<BASE_URL>`

This project is designed to be easy to run locally and easy to deploy.

<img width="1918" height="868" alt="image" src="https://github.com/user-attachments/assets/aaa924da-35e5-49e7-8a0c-d5f696826000" />


---

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Local Quickstart](#local-quickstart)
- [Deployment](#deployment)
  - [Backend on Render](#backend-on-render)
  - [Frontend on GitHub Pages](#frontend-on-github-pages)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [How Detection Works](#how-detection-works)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features
- Intrusion detection
  - Text analysis for phishing / SQLi / XSS / command injection / path traversal, etc.
  - CSV ingestion (scans all text fields)
  - Image OCR analysis (optional; disabled by default in Render deploy)
  - Confidence score and per-category matches
  - Compact UI charts per analysis
- Crypto
  - RSA and AES key generation
  - RSA test encrypt/decrypt and timings
  - AES key self-encrypt/decrypt demo and timings
- Analytics
  - Client-side demo charts (model comparison, ROC, timelines) with export hooks
  - Optional server-side report generation (enable `pandas`)
- Frontend
  - Static pages in `docs/`, host on GitHub Pages
  - Runtime backend selection via `?api=<BASE_URL>`
- Backend
  - Flask app under `Cyber/app_flask.py`
  - CORS enabled for cross-origin pages
  - Simple, dependency-light deployment

---

## Architecture

```
[Browser]
  └── GitHub Pages (docs/) – static HTML/JS
        └── calls → Flask API (Render)

[Flask Backend]
  - IDS endpoints
  - Crypto endpoints
  - Analytics/report endpoints

[Modules]
  - modules/simple_intrusion_detector.py   # rule-based IDS for text/CSV/image
  - modules/crypto_key_generator.py       # RSA/AES keys + timings
```

- Frontend and backend are decoupled. The frontend queries the backend base via `?api=<BASE_URL>`.
- CORS is enabled server-side so the Pages site can call Render URLs directly.

---

## Repository Layout

```
flaskcyber/
├─ Cyber/
│  ├─ app_flask.py                     # Flask app (main service)
│  ├─ requirements_flask.txt          # Backend requirements
│  └─ modules/                        # IDS & Crypto modules
│     ├─ simple_intrusion_detector.py
│     ├─ crypto_key_generator.py
│     └─ ... (other helpers)
│
├─ docs/                               # GitHub Pages frontend
│  ├─ index.html                       # Landing (enter API base)
│  ├─ ids.html                         # Intrusion Detection UI
│  ├─ crypto.html                      # Crypto UI (iframe wrapper)
│  └─ analytics.html                   # Analytics UI (iframe wrapper)
│
├─ render.yaml                         # (optional) Render blueprint
├─ .gitignore
└─ README.md
```

---

## Local Quickstart

Prerequisites: Python 3.12+

```bash
# 1) Create venv
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

# 2) Install backend deps
pip install -r Cyber/requirements_flask.txt

# 3) Run Flask backend
python Cyber/app_flask.py
# Server runs on http://127.0.0.1:5000

# 4) Open the frontend
#   - Open docs/index.html and paste http://127.0.0.1:5000
#   - Or go directly to:
#       docs/ids.html?api=http://127.0.0.1:5000
#       docs/crypto.html?api=http://127.0.0.1:5000
#       docs/analytics.html?api=http://127.0.0.1:5000
```

Note: OCR packages are heavy and omitted by default for minimal installs. Image analysis will return a helpful message if OCR is unavailable.

---

## Deployment

### Backend on Render
- Minimal configuration (UI form):
  - Root Directory: leave empty
  - Build Command: `pip install -r Cyber/requirements_flask.txt`
  - Start Command: `python Cyber/app_flask.py`
  - Environment variable: `PYTHON_VERSION=3.12.3`
- First cold request on the free plan can be slow.

### Frontend on GitHub Pages
- GitHub → Settings → Pages → Deploy from a branch → `main` and folder `/docs`.
- Live site: `https://<your-username>.github.io/<repo>/`
- Pass your backend URL as `?api=...`:
  - IDS: `ids.html?api=https://<your-service>.onrender.com`
  - Crypto: `crypto.html?api=https://<your-service>.onrender.com`
  - Analytics: `analytics.html?api=https://<your-service>.onrender.com`

---

## Configuration
- CORS: enabled via `flask_cors` for any origin by default (suitable for demo). Restrict in production.
- Upload size: configured in `app_flask.py` (`MAX_CONTENT_LENGTH`).
- Secret key: `app.secret_key` is a demo value—replace for production.

---

## API Endpoints

Base URL: your backend (local or Render), e.g. `http://127.0.0.1:5000`.

- `GET /api/system-status` – health/status (cpu/mem, demo accuracy)
- IDS
  - `POST /api/analyze-text` – `{ text: "..." }` → intrusion result
  - `POST /api/analyze-csv` – form-data `file=<csv>` → intrusion result
  - `POST /api/analyze-image` – form-data `file=<image>` → OCR+intrusion (if OCR available)
- Crypto
  - `POST /crypto` – form-data (`key_type`, `key_size`, optional entropy) → keys + timings
  - `POST /api/generate-keys` – JSON batch generation (demo)
- Analytics/Export
  - `GET /api/export-csv` – metrics CSV (requires metrics collected)
  - `GET /api/generate-report` – HTML report (requires `pandas` installed)

All IDS endpoints return a compact result including: `is_intrusion`, `severity`, `confidence_score`, `threats_detected[]`.

---

## How Detection Works
- Rule-based matching across categories in `modules/simple_intrusion_detector.py`.
- Compiled regex over normalized text, weighted by match count and category count → confidence.
- Heuristics for single-word phishing (e.g., "phishing", "phish").
- Image OCR path: if EasyOCR is available, extract text then run the same analyzer.

---

## Troubleshooting
- Render build fails for heavy packages
  - The project intentionally avoids large OCR/ML wheels by default. Keep `easyocr/torch/opencv` out for quick deploys.
- `ModuleNotFoundError: rsa`
  - Ensure `rsa==4.9.1` is in `Cyber/requirements_flask.txt` (already included).
- Cross-origin issues from Pages
  - CORS is enabled in the backend. Verify you’re passing `?api=<BASE_URL>` to the pages.
- Slow first request
  - Render free instances cold-start. Try again after ~30–60s.

---

## License
This project is for educational/demo purposes. Add a license of your choice before production use.
