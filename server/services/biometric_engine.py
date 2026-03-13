# Import sqlite3 FIRST to bind to the system libsqlite3 before cv2 overrides it
import sqlite3
# Import torch BEFORE chromadb or deepface to prevent TF/ONNX from shadowing PyTorch CUDA libs
import torch
import torchvision
import chromadb

from deepface import DeepFace
import numpy as np
import os
import torch
import cv2
import warnings
import threading
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from db.cruds.crud_student import save_student_vector, search_student_by_vector

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Device flags
# ---------------------------------------------------------------------------
DEVICE   = '0' if torch.cuda.is_available() else 'cpu'
USE_HALF = False # Previene CUDA misaligned address en la GTX 1050 Ti

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

from services.gpu_mutex import global_gpu_lock

print("[INFO][BiometricEngine] Warming up YOLO-Face en GPU...")
_dummy_face = np.zeros((320, 180, 3), dtype=np.uint8)
with global_gpu_lock:
    yolo_face_model(_dummy_face, device=DEVICE, verbose=False, imgsz=320, half=USE_HALF)
    yolo_face_model.track(_dummy_face, device=DEVICE, verbose=False, imgsz=320,
                          half=USE_HALF, persist=True)
    if DEVICE.startswith("0") or DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
print("[INFO][BiometricEngine] ✅ YOLO-Face listo.")

# Warmup DeepFace
print("[INFO][BiometricEngine] Warming up DeepFace (ArcFace)...")
try:
    DeepFace.represent(img_path=_dummy_face, model_name="ArcFace", detector_backend="skip", enforce_detection=False)
    print("[INFO][BiometricEngine] ✅ DeepFace ArcFace listo.")
except Exception as e:
    print(f"[ERROR][BiometricEngine] Error en warmup de DeepFace: {e}")

# ---------------------------------------------------------------------------
# Threshold de reconocimiento facial
#
# DeepFace ArcFace con métrica Coseno recomienda distancia < 0.68.
# Para embeddings normalizados (L2=1), ChromaDB (distancia L2 cuadrada) = 2 * (1 - cos).
# Coseno dist ~0.45 -> ChromaDB = 2 * (0.45) = 0.90. (Muy estricto para evitar falsos pos).
# ---------------------------------------------------------------------------
FACE_THRESHOLD = 0.70
FACE_EMBEDDING_VERSION = "deepface_arcface"
FACE_EMBEDDING_DTYPE = "float32"
FACE_REGISTER_JITTERS = int(os.getenv("FACE_REGISTER_JITTERS", "1"))
FACE_SOCKET_JITTERS = int(os.getenv("FACE_SOCKET_JITTERS", "1"))


def track_faces_yolo(image_rgb):
    """
    Canal rápido: detección + tracking de caras con ByteTrack (persist=True).
    Devuelve lista de {"location": (top,right,bottom,left), "track_id": int|None}.
    """
    with global_gpu_lock:
        results = yolo_face_model.track(
            image_rgb, device=DEVICE, verbose=False,
            imgsz=320, half=USE_HALF, persist=True
        )
        if DEVICE.startswith("0") or DEVICE.startswith("cuda"):
            torch.cuda.synchronize()
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
    def _serialize_face_vector(vector):
        return np.asarray(vector, dtype=np.float32).tolist()

    @staticmethod
    def _build_face_master_vector(vectors_buffer):
        stacked = np.asarray(vectors_buffer, dtype=np.float32)
        mean_vec = np.mean(stacked, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        return mean_vec.astype(np.float32).tolist()

    @staticmethod
    def _get_face_locations_yolo(image_rgb):
        """
        Detección pura de caras sin tracking (pipeline lento y registro).
        FP16 activo en GPU. imgsz=320 fijo.
        """
        with global_gpu_lock:
            results = yolo_face_model(
                image_rgb, device=DEVICE, verbose=False, imgsz=320, half=USE_HALF
            )
            if DEVICE.startswith("0") or DEVICE.startswith("cuda"):
                torch.cuda.synchronize()
        face_locations = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_locations.append((y1, x2, y2, x1))
        return face_locations
        
    @staticmethod
    def _extract_arcface(image_rgb, box):
        top, right, bottom, left = box
        h, w = image_rgb.shape[:2]
        
        # Expandir la caja ligeramente un 10% para ArcFace (mejora precision)
        bw, bh = right - left, bottom - top
        padding_x, padding_y = int(bw * 0.1), int(bh * 0.1)
        
        etop, ebottom = max(0, top - padding_y), min(h, bottom + padding_y)
        eleft, eright = max(0, left - padding_x), min(w, right + padding_x)

        face_crop = image_rgb[etop:ebottom, eleft:eright]
        if face_crop.size == 0:
            return None
            
        try:
            # DeepFace.represent puede tomar NumPy arrays en BGR.
            face_crop_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            rep = DeepFace.represent(img_path=face_crop_bgr, model_name="ArcFace", detector_backend="skip", enforce_detection=False)
            emb = np.array(rep[0]["embedding"], dtype=np.float32)
            # Normalizar a L2 unitario
            norm = np.linalg.norm(emb)
            if norm > 0:
                return (emb / norm).tolist()
        except Exception as e:
            print(f"[ERROR] Extrayendo ArcFace: {e}")
        return None

    @staticmethod
    def register_face(student_id: str, full_name: str, images_list: list):
        vectors_buffer = []
        for image_file in images_list:
            try:
                # cargar en RGB usando cv2
                image = cv2.imread(image_file)
                if image is None: continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                face_locations = BiometricEngine._get_face_locations_yolo(image_rgb)
                if len(face_locations) == 1:
                    encoding = BiometricEngine._extract_arcface(image_rgb, face_locations[0])
                    if encoding:
                        vectors_buffer.append(encoding)
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

        master_vector = BiometricEngine._build_face_master_vector(vectors_buffer)
        save_student_vector(student_id, master_vector, {
            "student_id": student_id,
            "full_name": full_name,
            "embedding_version": FACE_EMBEDDING_VERSION,
            "dtype": FACE_EMBEDDING_DTYPE,
            "normalized": True,
            "num_jitters": FACE_REGISTER_JITTERS,
            "source": "batch_register",
        })
        print(f"✅ Rostro registrado: '{full_name}' ({student_id})")
        return True

    def register_face_from_socket(self, student_id: str, full_name: str, images_list: list):
        vectors_buffer = []
        print(f"[SOCKET-IA] Procesando biométrico ArcFace para {full_name}...")
        for i, image_np in enumerate(images_list):
            try:
                # La imagen ya viene en RGB (desde PIL Image)
                face_locations = BiometricEngine._get_face_locations_yolo(image_np)
                if len(face_locations) == 1:
                    encoding = BiometricEngine._extract_arcface(image_np, face_locations[0])
                    if encoding:
                        vectors_buffer.append(encoding)
                        print(f"   ✅ Imagen {i+1}: vector ArcFace extraído OK.")
                    else:
                        print(f"   ❌ Imagen {i+1}: fallo de extracción DeepFace.")
                else:
                    print(f"   ⚠️  Imagen {i+1}: {len(face_locations)} rostros.")
            except Exception as e:
                print(f"   ❌ Imagen {i+1}: {e}")

        if not vectors_buffer:
            return False

        master_vector = BiometricEngine._build_face_master_vector(vectors_buffer)
        try:
            save_student_vector(student_id, master_vector, {
                "student_id": student_id,
                "full_name":  full_name,
                "embedding_version": FACE_EMBEDDING_VERSION,
                "dtype": FACE_EMBEDDING_DTYPE,
                "normalized": True,
                "num_jitters": FACE_SOCKET_JITTERS,
                "source":     "socket_bidirectional",
            })
            return True
        except Exception as e:
            print(f"❌ Error al guardar: {e}")
            return False

    @staticmethod
    def recognize_faces_in_frame(frame_rgb):
        """
        Reconoce caras: YOLO detecta rostros, ArcFace extrae características 512D.
        Se compara en ChromaDB con Distancia Euclidiana estricta L2 para evitar Desconocidos.
        """
        face_locations = BiometricEngine._get_face_locations_yolo(frame_rgb)
        if not face_locations:
            return []

        results = []
        for box in face_locations:
            top, right, bottom, left = box
            identity   = "Desconocido"
            student_id = None
            color      = (0, 0, 255)
            confidence = 0.0

            face_vector = BiometricEngine._extract_arcface(frame_rgb, box)
            
            if face_vector:
                match = search_student_by_vector(face_vector)
                if match and match["distance"] < FACE_THRESHOLD:
                    identity   = match["metadata"].get("full_name", "Estudiante")
                    student_id = match["metadata"].get("student_id", match.get("student_id"))
                    color      = (0, 255, 0)
                    confidence = round(max(0, (2.0 - match["distance"]) / 2.0) * 100, 1)

            results.append({
                "location":   (top, right, bottom, left),
                "identity":   identity,
                "student_id": student_id,
                "color":      color,
                "confidence": f"{confidence}%" if confidence > 0.0 else "N/A",
            })

        return results
