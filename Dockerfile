# Usamos una versión estable de Python sobre Linux
FROM python:3.10-slim-bookworm

# Instalar dependencias del sistema
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

# Establecemos el directorio de trabajo
WORKDIR /app

RUN pip install git+https://github.com/ageitgey/face_recognition_models

# Copiamos los archivos de dependencias
COPY requirements.txt .

# Instalamos las dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código
COPY . .

# Exponemos el puerto
EXPOSE 8000

# Comando para iniciar
CMD ["python", "main.py"]