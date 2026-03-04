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

    # Detectar TODOS los rostros en el frame
    students_detected = engine.recognize_faces_in_frame(frame)
    if not students_detected:
        return DetectResponse(status="No hay rostros reconocidos", students=[])

    # Detectar TODOS los cuerpos en el frame
    cuerpos = ClothingEngine.extract_all_torsos(frame)
    
    resultados_finales = []

    # Unir cada Rostro con su respectivo Cuerpo
    for student in students_detected:
        top, right, bottom, left = student["location"]
        # Calcular el centro del rostro (Punto X, Y)
        face_x = (left + right) // 2
        face_y = (top + bottom) // 2

        has_uniform = False
        distancia = 999.0

        # Buscar en qué cuerpo encaja este rostro
        for cuerpo in cuerpos:
            bx1, by1, bx2, by2 = cuerpo["box"]
            
            # Si el rostro está dentro del cuadro de este cuerpo, le evaluamos la ropa
            if bx1 <= face_x <= bx2 and by1 <= face_y <= by2:
                torso = cuerpo["torso"]
                clothing_vector = ClothingEngine.get_clothing_embedding(torso)
                uniforme_detectado = search_uniform_by_vector(clothing_vector)
                
                if uniforme_detectado:
                    distancia = uniforme_detectado["distance"]
                    has_uniform = bool(distancia < 0.25) # Cambia esto según tu calibración actual
                
                break # Ya evaluamos la ropa de este alumno, pasamos al siguiente
        
        # Inyectar los resultados de ropa al diccionario del estudiante
        student["has_uniform"] = has_uniform
        student["clothing_distance"] = distancia
        resultados_finales.append(student)

    return DetectResponse(
        status="Alumnos procesados",
        students=resultados_finales
    )