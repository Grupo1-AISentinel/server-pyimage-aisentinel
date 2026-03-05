from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np
from api.schemas import StudentRegister, DetectResponse, UniformRegister

from services.biometric_engine import BiometricEngine
from services.clothing_engine import ClothingEngine

router = APIRouter()
engine = BiometricEngine()

@router.get("/health")
def health_check():
    return {"status": "PYIMAGE Server is healthy", "version": "1.0.0"}

@router.post("/register")
def register_student(datos: StudentRegister):
    return {"message": f"Estudiante {datos.card} listo para procesar"}

@router.post("/register/uniform")
def register_uniform(datos: UniformRegister):
    success = ClothingEngine.register_clothing_item(datos.item_id, datos.item_type, datos.images)
    if success:
        return {"message": f"Prenda '{datos.item_type}' ({datos.item_id}) registrada exitosamente en ChromaDB."}
    return {"error": "No se pudo registrar la prenda."}

@router.post("/api/detect", response_model=DetectResponse)
async def detect_student(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Detectar TODOS los rostros en el frame con altisima precisión
    students_detected = engine.recognize_faces_in_frame(frame)
    if not students_detected:
        return DetectResponse(status="No hay rostros reconocidos", students=[])

    # Detectar TODOS los cuerpos en el frame mediante YOLO
    cuerpos = ClothingEngine.extract_all_torsos(frame)
    
    resultados_finales = []

    # Unir cada Rostro con su respectivo Cuerpo
    for student in students_detected:
        top, right, bottom, left = student["location"]
        # Calcular el centro del rostro (Punto X, Y)
        face_x = (left + right) // 2
        face_y = (top + bottom) // 2

        has_uniform = False
        clothing_details = "Rostro sin cuerpo visible"

        # Buscar en qué cuerpo encaja este rostro
        for cuerpo in cuerpos:
            bx1, by1, bx2, by2 = cuerpo["box"]
            
            # Si el centro del rostro está dentro del cuadro de este cuerpo entero
            if bx1 <= face_x <= bx2 and by1 <= face_y <= by2:
                # Ya tenemos a la persona validada, revisamos la ropa que lleva usando el modelo especial
                crop = cuerpo["crop"]
                has_uniform, clothing_details = ClothingEngine.validate_uniform(crop)
                break # Ya evaluamos la ropa de este alumno, pasamos al siguiente
        
        # Inyectar los resultados de ropa al diccionario del estudiante
        student["has_uniform"] = has_uniform
        student["clothing_details"] = clothing_details
        resultados_finales.append(student)

    return DetectResponse(
        status="Alumnos procesados",
        students=resultados_finales
    )