# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# gcc / g++ are required to compile certain torch and numpy C extensions.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching.
COPY requirements.txt ./

# Install PyTorch CPU-only build from the official CPU index.
# This avoids the much larger CUDA build (~2 GB) — CPU build is ~200 MB.
RUN pip install --no-cache-dir \
        torch \
        --index-url https://download.pytorch.org/whl/cpu

# Install the Chronos-T5 inference library.
# The standalone ``chronos-forecasting`` package is lighter than AutoGluon.
RUN pip install --no-cache-dir chronos-forecasting

# Install the rest of the project dependencies (including FastAPI, uvicorn,
# transformers, accelerate, etc.) from PyPI.
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
# Copy everything except data/ (mounted as a volume at runtime).
COPY regime/ ./regime/
COPY main_regime.py ./

# ── Runtime configuration ─────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    REGIME_HOST=0.0.0.0 \
    REGIME_PORT=8000

EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=120s \
    --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["uvicorn", "main_regime:app", "--host", "0.0.0.0", "--port", "8000"]
