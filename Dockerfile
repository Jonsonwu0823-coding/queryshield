# QueryShield service image (B4b).
#
# The project directory layout is kept on purpose: the product finds its data files relative
# to the source tree (src/queryshield/... -> fixtures/), so src/ and fixtures/ must sit
# side by side, and the package is installed in editable mode.
#
# Only what the service, the one-shot setup step and the demo scripts need is copied.
# tests/, evals/ (development evaluation data), docs/ and every .env file stay out: see
# .dockerignore, which is a second line of defence for the same rule.
#
# The base image is pinned by tag AND by the digest of its multi-architecture index,
# so amd64 and arm64 get the same, reproducible content.
FROM python:3.14.3-slim-bookworm@sha256:f21c0d5a44c56805654c15abccc1b2fd576c8d93aca0a3f74b4aba2dc92510e2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Exact third-party versions first (pywin32 has a platform marker and is skipped on Linux),
# then the project itself without resolving dependencies again.
COPY requirements.lock pyproject.toml ./
RUN pip install -r requirements.lock
COPY src ./src
RUN pip install --no-deps -e .

COPY fixtures ./fixtures
COPY migrations ./migrations
COPY scripts/setup_databases.py scripts/bootstrap_db.py scripts/bootstrap_demo_db.py scripts/generate_demo_data.py \
     scripts/demo_run.py scripts/demo_walkthrough.py scripts/b2b_http_smoke.py scripts/new_env.py ./scripts/

# The only writable place besides /tmp: the run/approval store and the model call records.
RUN mkdir -p /var/lib/queryshield && chown app:app /var/lib/queryshield

USER app
EXPOSE 8000

# Standard library only: no curl in the image.
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

# One process: two processes sharing the state store can execute one approval twice (docs/operations.md).
CMD ["python", "-m", "uvicorn", "queryshield.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
