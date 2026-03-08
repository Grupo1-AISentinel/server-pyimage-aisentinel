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

def process_frame_fast_from_bytes(img_bytes: bytes) -> list:
    """
    Detección rápida: solo YOLO face (~50-80ms en CPU).
    No corre face_recognition.face_encodings ni ResNet ni ChromaDB.
    Devuelve únicamente las posiciones de caras detectadas.
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    from services.biometric_engine import BiometricEngine
    face_locations = BiometricEngine._get_face_locations_yolo(frame)
    return [{"location": list(loc)} for loc in face_locations]

def process_frame_from_bytes(img_bytes: bytes, biometric_engine) -> list:
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return None
        
    return process_frame_logic(frame, biometric_engine)

def process_frame_logic(frame: np.ndarray, biometric_engine) -> list:
    import concurrent.futures

    # Ejecutar detección de rostros y de cuerpos en PARALELO.
    # Ambas operaciones son completamente independientes entre sí.
    # Tiempo total = max(T_rostros, T_cuerpos) en lugar de T_rostros + T_cuerpos.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_faces = executor.submit(biometric_engine.recognize_faces_in_frame, frame)
        future_bodies = executor.submit(ClothingEngine.extract_all_torsos, frame)
        students_detected = future_faces.result()
        cuerpos = future_bodies.result()

    if not students_detected:
        return []

    resultados_finales = []

    # Unir cada Rostro con su respectivo Cuerpo
    for student in students_detected:
        top, right, bottom, left = student["location"]
        # Calcular el centro del rostro (Punto X, Y)
        face_x = (left + right) // 2
        face_y = (top + bottom) // 2

        has_uniform = False
        clothing_details = "Rostro sin cuerpo visible"
        needs_full_body_view = False
        clothing_boxes = []

        # Buscar en qué cuerpo encaja este rostro
        for cuerpo in cuerpos:
            bx1, by1, bx2, by2 = cuerpo["box"]
            
            # Si el centro del rostro está dentro del cuadro de este cuerpo entero
            if bx1 <= face_x <= bx2 and by1 <= face_y <= by2:
                # Ya tenemos a la persona validada, revisamos la ropa que lleva usando el modelo especial
                crop = cuerpo["crop"]
                # La tupla devuelve: (valido, log, alerta_piernas, cajas_ropa_relativas)
                has_uniform, clothing_details, needs_full_body_view, r_clothing_boxes = ClothingEngine.validate_uniform(crop)
                
                # Convertir coordenadas relativas del crop a coordenadas absolutas de la imagen para el frontend
                for cb in r_clothing_boxes:
                    cx1, cy1, cx2, cy2 = cb["box"]
                    clothing_boxes.append({
                        "class": cb["class"],
                        "box": [bx1 + cx1, by1 + cy1, bx1 + cx2, by1 + cy2],
                        "valid": cb.get("valid", False)
                    })

                break # Ya evaluamos la ropa de este alumno, pasamos al siguiente
        
        # Inyectar los resultados de ropa al diccionario del estudiante
        student["has_uniform"] = has_uniform
        student["clothing_details"] = clothing_details
        student["needs_full_body_view"] = needs_full_body_view
        student["clothing_boxes"] = clothing_boxes
        resultados_finales.append(student)

    return resultados_finales

@router.post("/api/detect", response_model=DetectResponse)
async def detect_student(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    resultados_finales = process_frame_logic(frame, engine)
    
    if not resultados_finales:
        return DetectResponse(status="No hay rostros reconocidos", students=[])

    return DetectResponse(
        status="Alumnos procesados",
        students=resultados_finales
    )