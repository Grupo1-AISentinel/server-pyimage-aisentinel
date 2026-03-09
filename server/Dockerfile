# syntax=docker/dockerfile:1
# Usamos una versión estable de Python sobre Linux
FROM python:3.10-slim-bookworm

# Instalar dependencias del sistema
# Esta capa se cachea y no se re-ejecuta a menos que cambies los paquetes de apt
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch con soporte CUDA 11.7 — compatible con Pascal (sm_61) como GTX 1050 Ti.
# Capa separada: solo se re-ejecuta si cambias la versión de torch aquí.
# Si cambias a GPU Volta+ (RTX 2000/3000/4000), usa torch>=2.1.0 con cu118/cu121.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    torch==2.0.1+cu117 \
    torchvision==0.15.2+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# face_recognition_models se instala desde GitHub (pesos de dlib).
# Capa separada: se cachea mientras no cambie el commit del repo.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install git+https://github.com/ageitgey/face_recognition_models

# Copiar requirements ANTES del código para aprovechar el caché:
# Si solo cambias .py → las capas de pip se reutilizan (instantáneo).
# Si cambias requirements.txt → solo re-ejecuta pip install (sin re-compilar dlib).
# dlib requiere cmake como paquete Python. pip usa build isolation por defecto,
# lo que crea un entorno limpio donde cmake no está disponible.
# --no-build-isolation desactiva ese aislamiento para que dlib vea cmake.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip cmake && \
    pip install dlib --no-build-isolation

COPY requirements.txt .
# face_recognition depende de dlib (ya instalado arriba → pip lo reutiliza sin compilar)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copiar el código del servidor (última capa — la más frecuentemente cambiada)
COPY server/ .

EXPOSE 8000
CMD ["python", "main.py"]
