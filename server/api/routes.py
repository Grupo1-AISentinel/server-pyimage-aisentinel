from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from api.schemas import StudentRegister, DetectResponse, UniformRegister

from services.biometric_engine import BiometricEngine
from services.clothing_engine   import ClothingEngine

router = APIRouter()
engine = BiometricEngine()

# ---------------------------------------------------------------------------
# ThreadPoolExecutor persistente (module-level)
#
# Crear un ThreadPoolExecutor por cada request (~cada 1.5 s) tiene overhead
# no trivial: Python crea/destruye hilos del SO en cada ciclo.
# Con un pool persistente, los hilos se reutilizan → sin latencia de creación.
#
# max_workers=4: dos tareas siempre en paralelo (faces + bodies) con dos hilos
# extra para la validación paralela de ropa si hay múltiples personas.
# ---------------------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=4)


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
        return {"message": f"Prenda '{datos.item_type}' ({datos.item_id}) registrada."}
    return {"error": "No se pudo registrar la prenda."}


def process_frame_fast_from_bytes(img_bytes: bytes) -> list:
    """
    Canal rápido: YOLO face con ByteTrack (~30-80 ms en GPU con FP16).
    Devuelve lista de {"location": [...], "track_id": int|None}.
    El track_id persiste frame a frame para la misma cara → el cliente lo usa
    como clave de matching en su Kalman filter, eliminando ambigüedad cuando
    hay múltiples personas en escena.
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    from services.biometric_engine import track_faces_yolo
    return [{"location": list(f["location"]), "track_id": f["track_id"]}
            for f in track_faces_yolo(frame)]


def process_frame_from_bytes(img_bytes: bytes, biometric_engine) -> list:
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    return process_frame_logic(frame, biometric_engine)


def _validate_person(student, cuerpos):
    """
    Asocia un estudiante detectado con su cuerpo más cercano y valida su uniforme.
    Diseñado para ejecutarse en paralelo con _executor cuando hay múltiples personas.

    Optimización: si la identidad es "Desconocido" se omite la validación de
    uniforme (no tiene sentido gastar YOLO+ResNet en alguien no registrado).
    """
    top, right, bottom, left = student["location"]
    face_x = (left + right) // 2
    face_y = (top  + bottom) // 2

    student["has_uniform"]          = False
    student["clothing_details"]     = "Rostro sin cuerpo visible"
    student["needs_full_body_view"] = False
    student["clothing_boxes"]       = []

    # No gastar recursos en personas desconocidas
    if student.get("identity") == "Desconocido":
        student["clothing_details"] = "No registrado"
        return student

    for cuerpo in cuerpos:
        bx1, by1, bx2, by2 = cuerpo["box"]
        if bx1 <= face_x <= bx2 and by1 <= face_y <= by2:
            crop = cuerpo["crop"]
            has_uniform, details, needs_body, r_boxes = ClothingEngine.validate_uniform(crop)

            # Convertir coords relativas del crop → absolutas de imagen
            clothing_boxes = [
                {
                    "class": cb["class"],
                    "box":   [bx1 + cb["box"][0], by1 + cb["box"][1],
                              bx1 + cb["box"][2], by1 + cb["box"][3]],
                    "valid": cb.get("valid", False),
                }
                for cb in r_boxes
            ]

            student["has_uniform"]          = has_uniform
            student["clothing_details"]     = details
            student["needs_full_body_view"] = needs_body
            student["clothing_boxes"]       = clothing_boxes
            break

    return student


def process_frame_logic(frame: np.ndarray, biometric_engine) -> list:
    """
    Pipeline completo de detección:

    Fase 1 (paralela):
      - Tarea A: YOLO-Face → face_recognition encoding → ChromaDB (con identity cache)
      - Tarea B: YOLO-Person → recortes de cuerpo

    Fase 2 (paralela por persona):
      - Para cada estudiante detectado: YOLO-Clothing → ResNet → ChromaDB
      - Si hay N personas, las N validaciones de ropa se ejecutan en paralelo.

    El uso del _executor persistente evita el overhead de crear/destruir un
    ThreadPoolExecutor en cada llamada (~4-8 ms por ciclo eliminados).
    """
    # --- Fase 1: detección biométrica y de cuerpos en paralelo ---
    future_faces  = _executor.submit(biometric_engine.recognize_faces_in_frame, frame)
    future_bodies = _executor.submit(ClothingEngine.extract_all_torsos, frame)

    students_detected = future_faces.result()
    cuerpos           = future_bodies.result()

    if not students_detected:
        return []

    if len(students_detected) == 1:
        # Optimización: una sola persona → sin overhead de submit a executor
        return [_validate_person(students_detected[0], cuerpos)]

    # --- Fase 2: validación de ropa en paralelo para múltiples personas ---
    # Si 3 personas son detectadas, las 3 validaciones de uniforme corren
    # simultáneamente en lugar de secuencialmente.
    # Tiempo total = max(T_persona_1, T_persona_2, ...) en lugar de ΣT.
    futures = {
        _executor.submit(_validate_person, student, cuerpos): i
        for i, student in enumerate(students_detected)
    }

    resultados = [None] * len(students_detected)
    for fut in as_completed(futures):
        idx = futures[fut]
        try:
            resultados[idx] = fut.result()
        except Exception as e:
            resultados[idx] = students_detected[idx]  # fallback sin ropa
            print(f"[ERROR] validate_person idx={idx}: {e}")

    return [r for r in resultados if r is not None]


@router.post("/api/detect", response_model=DetectResponse)
async def detect_student(file: UploadFile = File(...)):
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    resultados_finales = process_frame_logic(frame, engine)
    if not resultados_finales:
        return DetectResponse(status="No hay rostros reconocidos", students=[])
    return DetectResponse(status="Alumnos procesados", students=resultados_finales)
