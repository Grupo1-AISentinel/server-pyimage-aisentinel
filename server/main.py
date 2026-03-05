from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services.biometric_engine import BiometricEngine 
import uvicorn
import os
import logging
import threading 
import socketio  
import base64
import cv2
import numpy as np

# Importar las rutas desde la carpeta api
from api.routes import router as api_router 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AISentinel")

app = FastAPI(title="AISentinel")

# Configuración estricta de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

# Socket.io

@sio.event
def connect():
    logger.info("✅ Conectado bidireccionalmente con Node.js")

@sio.on('enviar_a_python')
def handle_node_data(data):
    """ Escucha el evento de registro de Node.js """
    try:
        nombre = data.get('nombre')
        carnet = data.get('carnet')
        fotos_raw = data.get('fotos', [])

        logger.info(f"📸 Recibidas {len(fotos_raw)} imágenes para el carnet: {carnet}")

        lista_imagenes = []
        for f in fotos_raw:
            img_bytes = base64.b64decode(f['base64'])
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                lista_imagenes.append(img)

        # Llamada a tu motor de IA
        # IMPORTANTE: Asegúrate que process_registration maneje la lógica de guardado
        success = engine.process_registration(carnet, lista_imagenes)

        if success:
            sio.emit('python_registro_completado', {'carnet': carnet, 'status': 'success'})
            logger.info(f"🎯 Registro biométrico completado para {carnet}")

    except Exception as e:
        logger.error(f"❌ Error procesando datos de Node: {e}")

def start_socket():
    """ Función para conectar el socket en un hilo aparte """
    try:
        # Reemplaza con la URL real de tu servidor Node
        sio.connect("http://localhost:3000") 
        sio.wait()
    except Exception as e:
        logger.error(f"No se pudo conectar a Node.js: {e}")

@app.on_event("startup")
async def startup_event():
    logger.info("🟢 Servidor AI Sentinel Iniciado...")

@app.get("/")
def read_root():
    return {"status": "online", "service": "AISentinel Facial Recognition API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)