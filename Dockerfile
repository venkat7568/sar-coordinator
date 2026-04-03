# SAR Coordinator Environment — Production Dockerfile
# Builds the FastAPI environment server for HuggingFace Spaces deployment.
# Works standalone — no dependency on meta-pytorch base image.
#
# Build:  docker build -t sar-coordinator .
# Run:    docker run -p 8000:8000 sar-coordinator

FROM python:3.11-slim

WORKDIR /app

# System deps — curl for healthcheck, git for any VCS pip installs
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache friendly)
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY . /app/env

# PYTHONPATH so `from models import ...` and `from server.app import ...` both work
ENV PYTHONPATH="/app/env:/app/env/server"

# HuggingFace Spaces injects secrets as env vars — declare them here as defaults
# Override at runtime: docker run -e HF_TOKEN=... -e API_BASE_URL=...
ENV HF_TOKEN=""
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/schema || exit 1

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/env"]
