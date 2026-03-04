from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np
from api.schemas import StudentRegister, DetectResponse

from services.biometric_engine import BiometricEngine
from services.clothing_engine import ClothingEngine
from db.cruds.crud_uniform import search_uniform_by_vector

router = APIRouter()
engine = BiometricEngine()

@router.get("/health")
def health_check():
    return {"status": "PYIMAGE Server is healthy", "version": "1.0.0"}

@router.post("/register")
def register_student(datos: StudentRegister):
    return {"message": f"Estudiante {datos.card} listo para procesar"}

@router.post("/api/detect", response_model=DetectResponse)
async def detect_student(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Biometría
    student = engine.recognize_faces_in_frame(frame)
    if not student:
        return DetectResponse(status="No hay rostro reconocido")

    # Ropa (YOLO + ResNet)
    torso = ClothingEngine.extract_torso(frame)
    if torso is not None:
        clothing_vector = ClothingEngine.get_clothing_embedding(torso)
        
        # Consultar DB
        uniforme_detectado = search_uniform_by_vector(clothing_vector)
        
        has_uniform = False
        distancia = 999.0
        if uniforme_detectado:
            distancia = uniforme_detectado["distance"]
            has_uniform = bool(distancia < 15.0)

        return DetectResponse(
            status="Alumno detectado",
            student=student,
            has_uniform=has_uniform,
            clothing_distance=distancia
        )
    
    return DetectResponse(
        status="Rostro detectado, pero no se pudo ver el uniforme", 
        student=student
    )