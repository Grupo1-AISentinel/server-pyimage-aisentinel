# syntax=docker/dockerfile:1
FROM python:3.10-slim-bookworm

# 1. Dependencias base (sin herramientas pesadas de compilación)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    liblapack-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. PyTorch con CUDA 11.7 (Garantiza tu rendimiento en GPU)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    torch==2.0.1+cu117 \
    torchvision==0.15.2+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# 3. Dependencias del proyecto
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# 4. Código fuente
COPY server/ .

EXPOSE 8000
CMD ["python", "main.py"]