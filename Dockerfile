# ── GOV-AI 2.0 — Image Docker (Sprint 1) ─────────────────────────────────────
# Python 3.11-slim + Tesseract 5 + dépendances OCR/ML
# Déploiement CENADI 100% on-premise

FROM python:3.11-slim

LABEL maintainer="NJILLA TCHAGADICK NICOL EMMANUEL <njilla653@enspy.cm>"
LABEL version="2.0.0-sprint1"
LABEL description="GOV-AI 2.0 — Assistant gouvernemental intelligent (Cameroun)"

WORKDIR /app

# ── Dépendances système ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OCR
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    # Traitement d'images (libgl1 remplace libgl1-mesa-glx sur Debian >= Bullseye)
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    # Curl pour health checks
    curl \
    # Git (pour les modèles HuggingFace locaux)
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Dépendances Python ─────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Code applicatif ────────────────────────────────────────────────────────────
COPY . .

# ── Répertoires de stockage ────────────────────────────────────────────────────
RUN mkdir -p \
    /app/storage/documents \
    /app/storage/ocr_cache \
    /app/storage/model_cache \
    /app/logs \
    /app/eval/datasets \
    /app/eval/reports \
    /app/data \
    /app/metadata/chunks \
    /app/metadata/chat_history

# ── Utilisateur non-root (sécurité) ───────────────────────────────────────────
RUN useradd --no-create-home --shell /bin/false govai && \
    chown -R govai:govai /app
USER govai

# ── Ports ──────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# ── Démarrage ──────────────────────────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
