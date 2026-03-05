from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging

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

@app.on_event("startup")
async def startup_event():
    logger.info("🟢 Servidor AI Sentinel Iniciado...")

@app.get("/")
def read_root():
    return {"status": "online", "service": "AISentinel Facial Recognition API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)