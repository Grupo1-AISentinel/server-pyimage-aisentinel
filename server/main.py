from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging
import base64
import io
import time
import cv2
import numpy as np
import socketio
from threading import Thread
from PIL import Image, ImageOps


# Importar las rutas desde la carpeta api
from api.routes import router as api_router 

try:
    from services.biometric_engine import BiometricEngine
    logger_name = "AISentinel"
except ImportError:
    class BiometricEngine:
        def register_face(self, carnet, nombre, fotos):
            return False
    logger_name = "AISentinel-Debug"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(logger_name)
sio = socketio.Client(reconnection=True, reconnection_attempts=10, reconnection_delay=5)
engine = BiometricEngine()
app = FastAPI(title="AISentinel")

# Configuración estricta de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@sio.event
def connect():
    logger.info("✅ Conectado bidireccionalmente con Node.js (Puerto 3067)")

@sio.event
def disconnect():
    logger.info("❌ Desconectado de Node.js")

@sio.on('enviar_a_python')
def handle_node_data(data):
    carnet = data.get('carnet', 'Desconocido')
    nombre = data.get('nombre', 'Sin Nombre')
    fotos_raw = data.get('fotos', [])

    try:
        logger.info(f"📸 Procesando registro para: {nombre} ({carnet})")
        lista_imagenes = []

        for f in fotos_raw:
            try:
                raw_data = f.get('buffer') or f.get('base64')
                if not raw_data:
                    continue

                if isinstance(raw_data, str):
                    if "," in raw_data:
                        raw_data = raw_data.split(",")[1]
                    img_bytes = base64.b64decode(raw_data)
                else:
                    img_bytes = raw_data

                image = Image.open(io.BytesIO(img_bytes))
                image = ImageOps.exif_transpose(image)  # <--- Corrige la rotación automática
                image = image.convert('RGB')
                
                img_np = np.array(image)

                lista_imagenes.append(img_np)
                logger.info(f"   - Imagen decodificada: {image.size[0]}x{image.size[1]}")

            except Exception as e:
                logger.error(f"   - Error procesando imagen individual: {e}")

        if not lista_imagenes:
            raise ValueError("No se pudieron decodificar imágenes válidas.")

        success = engine.register_face_from_socket(carnet, nombre, lista_imagenes)

        if success:
            logger.info(f"🎯 Registro biométrico EXITOSO: {carnet}")
            sio.emit('python_registro_completado', {'carnet': carnet, 'status': 'success'})
        else:
            logger.error(f"❌ La IA no detectó rostros en las imágenes de {carnet}")
            sio.emit('python_registro_completado', {
                'carnet': carnet, 
                'status': 'error', 
                'message': 'No se detectó un rostro claro. Intente de nuevo.'
            })

    except Exception as e:
        logger.error(f"❌ Error crítico en el flujo: {e}")
        sio.emit('python_registro_completado', {'carnet': carnet, 'status': 'error', 'message': str(e)})

def start_socket():
    node_url = "http://host.docker.internal:3067"
    while True:
        if not sio.connected:
            try:
                logger.info(f"⏳ Intentando conectar a {node_url}...")
                sio.connect(node_url)
                sio.wait()
            except Exception:
                time.sleep(5)
        else:
            time.sleep(10)

app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    logger.info("🟢 Servidor AI Sentinel Iniciado...")
    thread = Thread(target=start_socket, daemon=True)
    thread.start()

@app.get("/")
def read_root():
    return {"status": "online", "service": "AISentinel Facial Recognition API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)