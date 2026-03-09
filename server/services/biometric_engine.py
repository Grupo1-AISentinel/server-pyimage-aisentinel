import face_recognition
import numpy as np
import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from db.cruds.crud_student import save_student_vector, search_student_by_vector

# ---------------------------------------------------------------------------
# Device flags
# ---------------------------------------------------------------------------
DEVICE   = '0' if torch.cuda.is_available() else 'cpu'
USE_HALF = torch.cuda.is_available()

if torch.cuda.is_available():
    _gpu_name = torch.cuda.get_device_name(0)
    _vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"[CUDA][BiometricEngine] ✅ GPU: {_gpu_name} | VRAM: {_vram_gb:.1f} GB | FP16={USE_HALF}")
else:
    print("[CUDA][BiometricEngine] ⚠️  Sin GPU — YOLO-Face en CPU")

# ---------------------------------------------------------------------------
# Carga y warmup del modelo YOLO face
# ---------------------------------------------------------------------------
print("[INFO][BiometricEngine] Cargando modelo YOLO11 Face (AdamCodd)...")
try:
    hf_model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
    yolo_face_model = YOLO(hf_model_path)
except Exception as e:
    print(f"[WARN][BiometricEngine] Fallo HF ({e}), usando yolo11n-face.pt local")
    yolo_face_model = YOLO('yolo11n-face.pt')

print("[INFO][BiometricEngine] Warming up YOLO-Face en GPU...")
_dummy_face = np.zeros((320, 180, 3), dtype=np.uint8)
yolo_face_model(_dummy_face, device=DEVICE, verbose=False, imgsz=320, half=USE_HALF)
yolo_face_model.track(_dummy_face, device=DEVICE, verbose=False, imgsz=320,
                      half=USE_HALF, persist=True)
print("[INFO][BiometricEngine] ✅ YOLO-Face listo.")

# ---------------------------------------------------------------------------
# Threshold de reconocimiento facial
#
# ChromaDB usa distancia L2 cuadrada.
# face_recognition recomienda 0.6 Euclidiano → L2 = 0.6² = 0.36.
# Usamos 0.32 (≈ 0.57 Euclidiano) — más estricto que el valor por defecto.
#
# Ajuste vs. el anterior 0.42:
#   0.42 (≈0.65 Eucl.) es el valor "permisivo" general de face_recognition.
#   A 0.32 solo pasan coincidencias con alta certeza → eliminamos los falsos
#   positivos donde el sistema confundía desconocidos con estudiantes.
#   Trade-off: puede haber más falsos negativos (estudiante → Desconocido)
#   cuando la calidad del frame es baja o el ángulo es muy oblicuo.
# ---------------------------------------------------------------------------
FACE_THRESHOLD = 0.32


def track_faces_yolo(image_rgb):
    """
    Canal rápido: detección + tracking de caras con ByteTrack (persist=True).
    Devuelve lista de {"location": (top,right,bottom,left), "track_id": int|None}.
    """
    results = yolo_face_model.track(
        image_rgb, device=DEVICE, verbose=False,
        imgsz=320, half=USE_HALF, persist=True
    )
    faces = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            try:
                track_id = int(box.id.item()) if box.id is not None else None
            except Exception:
                track_id = None
            faces.append({"location": (y1, x2, y2, x1), "track_id": track_id})
    return faces


class BiometricEngine:

    @staticmethod
    def _get_face_locations_yolo(image_rgb):
        """
        Detección pura de caras sin tracking (pipeline lento y registro).
        FP16 activo en GPU. imgsz=320 fijo.
        """
        results = yolo_face_model(
            image_rgb, device=DEVICE, verbose=False, imgsz=320, half=USE_HALF
        )
        face_locations = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_locations.append((y1, x2, y2, x1))
        return face_locations

    @staticmethod
    def register_face(student_id: str, full_name: str, images_list: list):
        vectors_buffer = []
        for image_file in images_list:
            try:
                image = face_recognition.load_image_file(image_file)
                face_locations = BiometricEngine._get_face_locations_yolo(image)
                if len(face_locations) == 1:
                    encoding = face_recognition.face_encodings(
                        image, face_locations, num_jitters=10
                    )[0]
                    vectors_buffer.append(encoding.tolist())
                elif len(face_locations) > 1:
                    print(f"[WARN] '{image_file}': múltiples rostros, omitida.")
                else:
                    print(f"[WARN] '{image_file}': sin rostro detectado, omitida.")
            except Exception as e:
                print(f"[ERROR] '{image_file}': {e}")

        if not vectors_buffer:
            raise ValueError("Sin imágenes válidas con rostro único.")
        if len(vectors_buffer) < 3:
            print("[WARN] Se recomienda ≥3 imágenes para mejor perfilamiento.")

        master_vector = np.mean(vectors_buffer, axis=0).tolist()
        save_student_vector(student_id, master_vector,
                            {"student_id": student_id, "full_name": full_name})
        print(f"✅ Rostro registrado: '{full_name}' ({student_id})")
        return True

    def register_face_from_socket(self, student_id: str, full_name: str, images_list: list):
        vectors_buffer = []
        print(f"[SOCKET-IA] Procesando biométrico para {full_name}...")
        for i, image_np in enumerate(images_list):
            try:
                face_locations = BiometricEngine._get_face_locations_yolo(image_np)
                if len(face_locations) == 1:
                    encoding = face_recognition.face_encodings(
                        image_np, face_locations, num_jitters=1
                    )[0]
                    vectors_buffer.append(encoding.tolist())
                    print(f"   ✅ Imagen {i+1}: rostro OK.")
                else:
                    print(f"   ⚠️  Imagen {i+1}: {len(face_locations)} rostros.")
            except Exception as e:
                print(f"   ❌ Imagen {i+1}: {e}")

        if not vectors_buffer:
            return False

        master_vector = np.mean(vectors_buffer, axis=0).tolist()
        try:
            save_student_vector(student_id, master_vector, {
                "student_id": student_id,
                "full_name":  full_name,
                "source":     "socket_bidirectional",
            })
            return True
        except Exception as e:
            print(f"❌ Error al guardar: {e}")
            return False

    @staticmethod
    def recognize_faces_in_frame(frame_rgb):
        """
        Reconoce caras en el frame: YOLO detecta ubicaciones, dlib extrae
        embeddings 128D, ChromaDB busca coincidencias con threshold estricto.

        Sin caché de identidad: cada frame se procesa desde cero.
        Esto garantiza que el resultado siempre corresponde exactamente al
        frame actual, sin riesgo de propagar identidades incorrectas.
        Threshold 0.32 (estricto) reduce falsos positivos.
        """
        face_locations = BiometricEngine._get_face_locations_yolo(frame_rgb)
        if not face_locations:
            return []

        # Batch encoding: dlib procesa todas las caras en una sola llamada
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        results = []
        for (top, right, bottom, left), face_vector in zip(face_locations, face_encodings):
            match      = search_student_by_vector(face_vector.tolist())
            identity   = "Desconocido"
            student_id = None
            color      = (0, 0, 255)
            confidence = 0.0

            if match and match["distance"] < FACE_THRESHOLD:
                identity   = match["metadata"].get("full_name", "Estudiante")
                student_id = match["metadata"].get("student_id", match.get("student_id"))
                color      = (0, 255, 0)
                confidence = round((1 - match["distance"]) * 100, 1)

            results.append({
                "location":   (top, right, bottom, left),
                "identity":   identity,
                "student_id": student_id,
                "color":      color,
                "confidence": f"{confidence}%",
            })

        return results
