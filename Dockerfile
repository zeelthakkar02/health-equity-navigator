# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Health Equity Navigator
#
# Credentials are NEVER baked into this image. The Vertex AI provider uses
# Application Default Credentials, supplied at run time (mounted ADC file
# locally, or the attached service account on Cloud Run / GKE).
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

WORKDIR /srv

# Dependency layer: copy only what the build backend needs so edits to app code
# do not invalidate the cached pip install.
COPY pyproject.toml README.md ./
COPY app ./app

RUN pip install --upgrade pip && pip install .

# Run unprivileged.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /srv
USER appuser

EXPOSE 8080

# Cloud Run ignores HEALTHCHECK and uses its own startup probe; this is for
# plain `docker run`.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD sh -c 'python -c "import os,urllib.request as u,sys; port=os.environ.get(\"PORT\",\"8080\"); sys.exit(0 if u.urlopen(f\"http://127.0.0.1:{port}/health\", timeout=3).status == 200 else 1)"'

# Shell form so ${PORT} expands: Cloud Run injects the port the container must
# listen on. exec replaces the shell so uvicorn stays PID 1 and receives SIGTERM.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
