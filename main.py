from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import logging

from services.biometric_engine import BiometricEngine
from core.seeder import seed_users

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AISentinel")

app = FastAPI(title="AISentinel")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- EVENTO DE INICIO
@app.on_event("startup")
async def startup_event():

    logger.info("Servidor Iniciando...")
    seed_users()
    # Mostrar un mensaje de bienvenida con datos del reconocimineto facial
    model = os.getenv("MODEL_FACE_RECOGNITION", "hog")
    logger.info(f"Modelo de Reconocimiento Facial: {model.upper() if model else 'HOG (por defecto)'}")
    threshold = os.getenv("FACE_RECOGNITION_THRESHOLD","sin especificar")
    logger.info(f"Umbral de Reconocimiento: {threshold}")

@app.get("/")
def read_root():
    return {"status": "online", "service": "AISentinel Facial Recognition API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)