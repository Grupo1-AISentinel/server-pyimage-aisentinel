# Import sqlite3 FIRST to bind to the system libsqlite3 before cv2 overrides it
import sqlite3
# Import torch BEFORE chromadb to prevent ONNXRuntime's bundled CUDA libraries from shadowing PyTorch's
import torch
import torchvision
import chromadb

import os
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO
import cv2
import numpy as np
import threading

from db.cruds.crud_uniform import (
    upsert_uniform_vector,
    search_uniform_by_vector_topk,
)

# ---------------------------------------------------------------------------
# Device flags
# ---------------------------------------------------------------------------
DEVICE_YOLO  = '0'    if torch.cuda.is_available() else 'cpu'
DEVICE_TORCH = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_HALF     = False # GTX 1050 Ti falla con HalfTensors (CUDA error: misaligned address)

SERVER_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SERVER_DIR.parent

UNIFORM_EMBEDDING_VERSION = "dinov2_small_v1"
UNIFORM_EMBEDDING_DTYPE = "float32"

STRUCTURAL_TO_LOGICAL_CLASS = {
    "jacket": "jacket",
    "jacket_open": "jacket",
    "jacket_close": "jacket",
    "shirt": "shirt",
    "pant": "pants",
    "pants": "pants",
    "accesory": "accesory",
}

JACKET_STATE_BY_CLASS = {
    "jacket_open": "open",
    "jacket_close": "close",
}

JACKET_DISPLAY_NAME_BY_CLASS = {
    "jacket": "CHUMPA",
    "jacket_open": "CHUMPA ABIERTA",
    "jacket_close": "CHUMPA CERRADA",
}


def _resolve_model_source(
    default_name: str,
    env_var: str,
    *,
    allow_registry_fallback: bool = False,
) -> str:
    override = os.getenv(env_var)
    candidates = []
    if override:
        override_path = Path(override)
        if override_path.is_absolute():
            candidates.append(override_path)
        else:
            candidates.extend([
                SERVER_DIR / override,
                REPO_ROOT / override,
            ])
    candidates.extend([
        SERVER_DIR / default_name,
        REPO_ROOT / default_name,
    ])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    if allow_registry_fallback:
        return default_name
    return str(SERVER_DIR / default_name)

if torch.cuda.is_available():
    _gpu_name = torch.cuda.get_device_name(0)
    _vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"[CUDA][ClothingEngine] ✅ GPU: {_gpu_name} | VRAM: {_vram_gb:.1f} GB | FP16={USE_HALF}")
else:
    print("[CUDA][ClothingEngine] ⚠️  Sin GPU — YOLO y DINOv2 en CPU")

# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
print("[INFO][ClothingEngine] Cargando modelos YOLO persona y ropa...")
yolo_person_model = YOLO(
    _resolve_model_source(
        "yolo26n.pt",
        "YOLO_PERSON_MODEL",
        allow_registry_fallback=True,
    )
)
yolo_clothing_structure_model = YOLO(
    _resolve_model_source("best.pt", "YOLO_CLOTHING_MODEL")
)

from services.gpu_mutex import global_gpu_lock

# DINOv2 como extractor de vectores robusto (384D)
print("[INFO][ClothingEngine] Cargando DINOv2 (facebook/dinov2-small)...")
dinov2_processor = None
dinov2_model = None
try:
    dinov2_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small', use_fast=False)
    dinov2_model = AutoModel.from_pretrained('facebook/dinov2-small').to(DEVICE_TORCH)
    dinov2_model.eval()
    print("[INFO][ClothingEngine] ✅ DINOv2 listo.")
except Exception as e:
    print(f"[ERROR][ClothingEngine] Falló la carga de DINOv2:\n{e}")

# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------
print("[INFO][ClothingEngine] Warming up modelos...")
_dummy_person   = np.zeros((320, 180, 3), dtype=np.uint8)
_dummy_clothing = np.zeros((200, 100, 3), dtype=np.uint8)

with global_gpu_lock:
    yolo_person_model(_dummy_person,    classes=[0], device=DEVICE_YOLO, verbose=False, imgsz=320, half=USE_HALF, conf=0.30)
    if DEVICE_YOLO.startswith("0") or DEVICE_YOLO.startswith("cuda"):
        torch.cuda.synchronize()
with global_gpu_lock:
    yolo_clothing_structure_model(_dummy_clothing,   device=DEVICE_YOLO, verbose=False, imgsz=320, half=USE_HALF, conf=0.30)
    if DEVICE_YOLO.startswith("0") or DEVICE_YOLO.startswith("cuda"):
        torch.cuda.synchronize()

try:
    with torch.no_grad():
        _dummy_tensor = torch.zeros(1, 3, 224, 224).to(DEVICE_TORCH)
        dinov2_model(_dummy_tensor)
except Exception:
    pass

print("[INFO][ClothingEngine] ✅ Sistemas de Validación listos.")

# ChromaDB cosine distance thresholds (1 - cosine_similarity).
# DINOv2 genera tensores muy densos.
# - Las camisas tienen logos formales que se deforman o tapan parcialmente, requiriendo un margen mayor (looser).
# - Los pantalones son tela lisa mayormente, por lo que otras telas del mismo color (pants de sudadera) 
#   pueden arrojar distancias muy bajas. Requerimos un margen más estricto (tighter).
DIST_THRESHOLD = {
    "jacket": 0.45,
    "shirt":  0.60, # Mucho más permisivo con camisas oficiales (con logos/dobleces)
    "pants":  0.24, # Punto intermedio para permitir el pantalón oficial pero bloquear deportivos
}
TOP_K_RESULTS = 5


class ClothingEngine:

    @staticmethod
    def _iou(box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union  = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _nms_clothing(items: list, iou_thresh: float = 0.40) -> list:
        if len(items) <= 1:
            return items
        sorted_items = sorted(
            items,
            key=lambda it: (
                it.get("priority", 0),
                (it["box"][2] - it["box"][0]) * (it["box"][3] - it["box"][1]),
            ),
            reverse=True,
        )
        kept  = [sorted_items[0]]
        for candidate in sorted_items[1:]:
            overlaps = any(
                ClothingEngine._iou(candidate["box"], k["box"]) > iou_thresh
                for k in kept
            )
            if not overlaps:
                kept.append(candidate)
        return kept

    @staticmethod
    def extract_all_torsos(frame):
        with global_gpu_lock:
            results = yolo_person_model(
                frame, classes=[0], device=DEVICE_YOLO,
                verbose=False, imgsz=320, half=USE_HALF
            )
            if DEVICE_YOLO.startswith("0") or DEVICE_YOLO.startswith("cuda"):
                torch.cuda.synchronize()
        personas = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size != 0:
                    personas.append({"box": (x1, y1, x2, y2), "crop": crop})
        return personas

    @staticmethod
    def _serialize_embedding(vector, *, normalize_l2: bool = False):
        if vector is None:
            return None
        arr = np.asarray(vector, dtype=np.float32)
        if normalize_l2:
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
        return arr.astype(np.float32).tolist()

    @staticmethod
    def _build_uniform_metadata(extra_meta: dict = None, source: str = "runtime"):
        metadata = {
            "valido": True,
            "embedding_version": UNIFORM_EMBEDDING_VERSION,
            "dtype": UNIFORM_EMBEDDING_DTYPE,
            "normalized": True,
            "source": source,
        }
        if extra_meta:
            metadata.update(extra_meta)
        return metadata

    @staticmethod
    def _infer_jacket_state(structural_class: str, metadata: dict = None):
        detected_state = JACKET_STATE_BY_CLASS.get(structural_class)
        if detected_state:
            return detected_state
        if metadata:
            return metadata.get("estado")
        return None

    @staticmethod
    def _get_embedding(cropped_image_bgr):
        """
        DINOv2 Embedding (384D). Highly robust to lighting, understands texture, pattern and color natively.
        APLICACIÓN DE CLAHE PARA SOMBRAS:
         - Convierte a LAB
         - Aplica ecualización adaptativa al canal L (luminancia) para eliminar sombras severas
         - Vuelve a convertir a RGB para mantener los pigmentos (A,B) intactos
        """
        if cropped_image_bgr is None or cropped_image_bgr.size == 0:
            return None

        # Eliminar sombras localizadas sin arruinar el color ("Advanced Technique")
        lab = cv2.cvtColor(cropped_image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        crop_clahe_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Procesador espera RGB
        color_rgb  = cv2.cvtColor(crop_clahe_bgr, cv2.COLOR_BGR2RGB)
        try:
            with global_gpu_lock:
                inputs = dinov2_processor(images=color_rgb, return_tensors="pt").to(DEVICE_TORCH)
                with torch.no_grad():
                    outputs = dinov2_model(**inputs)
                    # Forzar sincronización de VRAM antes de soltar el Mutex Global
                    if DEVICE_TORCH.startswith("cuda") or DEVICE_TORCH.startswith("0"):
                        torch.cuda.synchronize()
                    dinov2_vec = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            return ClothingEngine._serialize_embedding(dinov2_vec, normalize_l2=True)
        except Exception as e:
            print(f"[ERROR] Extrayendo DINOv2: {e}")
            return None

    @staticmethod
    def extract_all_clothing_from_image(frame_bgr, conf: float = 0.20) -> dict:
        with global_gpu_lock:
            results = yolo_clothing_structure_model(
                frame_bgr, device=DEVICE_YOLO, verbose=False,
                imgsz=640, half=USE_HALF, conf=conf
            )
            if DEVICE_YOLO.startswith("0") or DEVICE_YOLO.startswith("cuda"):
                torch.cuda.synchronize()
        detected: dict = {"jacket": [], "shirt": [], "pants": []}
        for result in results:
            for box in result.boxes:
                class_name = result.names[int(box.cls[0])]
                logical_name = STRUCTURAL_TO_LOGICAL_CLASS.get(class_name)
                if logical_name in detected:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame_bgr[y1:y2, x1:x2]
                    if crop.size > 0:
                        detected[logical_name].append(crop)
        return detected

    @staticmethod
    def register_clothing_item(item_id: str, item_type: str, images_list: list,
                               extra_meta: dict = None):
        yolo_classes = {
            "jacket": {"jacket", "jacket_open", "jacket_close"},
            "shirt": {"shirt"},
            "pants": {"pant", "pants"},
            "accesory": {"accesory"},
        }.get(item_type)
        if not yolo_classes:
            return False

        vectors_buffer = []
        for image_path in images_list:
            frame = cv2.imread(image_path)
            if frame is None:
                continue
            with global_gpu_lock:
                results = yolo_clothing_structure_model(
                    frame, device=DEVICE_YOLO, verbose=False, imgsz=640, half=USE_HALF
                )
                if DEVICE_YOLO.startswith("0") or DEVICE_YOLO.startswith("cuda"):
                    torch.cuda.synchronize()
            for result in results:
                for box in result.boxes:
                    if result.names[int(box.cls[0])] in yolo_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            v = ClothingEngine._get_embedding(crop)
                            if v:
                                vectors_buffer.append(v)

        if not vectors_buffer:
            print(f"[ERROR] YOLO no detectó '{item_type}' en las imágenes de {item_id}.")
            return False

        metadata = ClothingEngine._build_uniform_metadata(
            {
                "tipo": item_type,
                "base_id": item_id,
                **(extra_meta or {}),
            },
            source="api_register",
        )
        for idx, vector in enumerate(vectors_buffer):
            upsert_uniform_vector(f"{item_id}_{idx}", vector, metadata)
        return True

    @staticmethod
    def register_clothing_from_numpy(item_id: str, item_type: str,
                                     images_np: list, extra_meta: dict = None):
        """
        Registra prendas de uniforme a partir de imágenes numpy (BGR).

        Flujo:
          1. YOLO clothing structure detecta crops de la prenda en cada imagen.
          2. Para JACKET: usa JACKET_STATE_BY_CLASS para clasificar open/close
             y cada crop se guarda con su estado individual en metadata.
          3. Para TSHIRT/PANTS: los crops se guardan con su tipo directo.
          4. Cada crop → DINOv2 embedding 384D → upsert individual en ChromaDB.

        Retorna (success: bool, count: int).
        """
        # Mapear tipos de Node.js a tipos internos
        TYPE_MAP = {"JACKET": "jacket", "TSHIRT": "shirt", "PANTS": "pants"}
        internal_type = TYPE_MAP.get(item_type.upper(), item_type.lower())

        yolo_classes = {
            "jacket": {"jacket", "jacket_open", "jacket_close"},
            "shirt":  {"shirt"},
            "pants":  {"pant", "pants"},
        }.get(internal_type)

        if not yolo_classes:
            print(f"[ERROR] Tipo de uniforme desconocido: {item_type}")
            return False, 0

        vectors_saved = 0

        for img_idx, img_np in enumerate(images_np):
            if img_np is None or img_np.size == 0:
                continue

            # Asegurar BGR para YOLO (las imágenes de PIL llegan en RGB)
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = img_np

            with global_gpu_lock:
                results = yolo_clothing_structure_model(
                    frame_bgr, device=DEVICE_YOLO, verbose=False,
                    imgsz=640, half=USE_HALF, conf=0.20
                )
                if DEVICE_YOLO.startswith("0") or DEVICE_YOLO.startswith("cuda"):
                    torch.cuda.synchronize()

            for result in results:
                for box in result.boxes:
                    class_name = result.names[int(box.cls[0])]
                    if class_name not in yolo_classes:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame_bgr[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    vector = ClothingEngine._get_embedding(crop)
                    if vector is None:
                        continue

                    # Construir metadata individual por crop
                    crop_meta = {
                        "tipo": internal_type,
                        "base_id": item_id,
                        **(extra_meta or {}),
                    }

                    # Para chumpas: clasificar open/close por clase YOLO
                    if internal_type == "jacket":
                        estado = JACKET_STATE_BY_CLASS.get(class_name)
                        if estado:
                            crop_meta["estado"] = estado
                            print(f"  [REGISTER] img#{img_idx} → {class_name} → estado={estado}")
                        else:
                            # Clase genérica "jacket" sin estado específico
                            crop_meta["estado"] = "close"  # default conservador
                            print(f"  [REGISTER] img#{img_idx} → {class_name} → estado=close (default)")

                    metadata = ClothingEngine._build_uniform_metadata(
                        crop_meta, source="socket_register"
                    )
                    upsert_uniform_vector(
                        f"{item_id}_{vectors_saved}", vector, metadata
                    )
                    vectors_saved += 1

        if vectors_saved == 0:
            print(f"[ERROR] YOLO no detectó '{internal_type}' en las {len(images_np)} imágenes de {item_id}.")
        else:
            print(f"[OK] '{item_id}' ({internal_type}) — {vectors_saved} vectores guardados en ChromaDB.")

        return vectors_saved > 0, vectors_saved

    @staticmethod
    def _validate_garment_embedding(crop, tipo: str):
        """
        Validation vs ChromaDB specific dynamic uniform catalog.
        """
        # --- ANÁLISIS CV DE TEXTURA Y COLOR HSV PARA PANTALONES ---
        # Cortar el 50% central para aislar la tela y eliminar piso/fondo/zapatos de los bordes de la caja YOLO
        if tipo == "pants":
            h, w = crop.shape[:2]
            ch1, ch2 = int(h * 0.25), int(h * 0.75)
            cw1, cw2 = int(w * 0.25), int(w * 0.75)
            center_crop = crop[ch1:ch2, cw1:cw2]
            if center_crop.size > 0:
                hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
                v_median = np.median(hsv[:, :, 2])
                s_median = np.median(hsv[:, :, 1])
                print(f"    [COLOR_CHECK] Pants Center Median S={s_median:.1f}, V={v_median:.1f}")
                
                # Reglas estrictas:
                # 1. El negro y azul marino de pants deportivos absorben la luz (V cae por debajo de 40).
                # 2. Pantalones grises son un color desaturado oscuro, pero telas como jeans azules o pants de colores tienen S alto (>70).
                if v_median < 40.0:
                    return False, None, f"RECHAZADO SUDADERA/NEGRO(V={v_median:.1f})"
                if s_median > 75.0:
                    return False, None, f"RECHAZADO COLOR/JEAN(S={s_median:.1f})"

        vector = ClothingEngine._get_embedding(crop)
        if vector is None:
            return False, None, "embedding=None"

        topk = search_uniform_by_vector_topk(vector, tipo=tipo, k=TOP_K_RESULTS)
        if topk is None:
            return False, None, f"sin_ref({tipo})"

        median_dist = topk["median_distance"]
        best_dist   = topk["top_distance"]
        dist_lim    = DIST_THRESHOLD.get(tipo, 0.35)

        is_valid = median_dist < dist_lim

        debug = (
            f"best={best_dist:.3f} med={median_dist:.3f}(lim={dist_lim}) "
            f"-> {'VALID' if is_valid else 'REJECT'}"
        )
        return is_valid, topk["metadata"], debug

    @staticmethod
    def validate_uniform(person_crop):
        with global_gpu_lock:
            results = yolo_clothing_structure_model(
                person_crop, device=DEVICE_YOLO, verbose=False,
                imgsz=320, half=USE_HALF, conf=0.20
            )
            if DEVICE_YOLO.startswith("0") or DEVICE_YOLO.startswith("cuda"):
                torch.cuda.synchronize()

        detected = {
            "jacket":   [],
            "shirt":    [],
            "pants":    [],
            "accesory": [],
        }
        has_pants_structure = False

        for result in results:
            for box in result.boxes:
                class_name = result.names[int(box.cls[0])]
                logical_name = STRUCTURAL_TO_LOGICAL_CLASS.get(class_name)
                if logical_name is None:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = person_crop[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                if logical_name == 'jacket':
                    detected["jacket"].append({
                        "crop": crop, "box": [x1, y1, x2, y2],
                        "display_name": JACKET_DISPLAY_NAME_BY_CLASS.get(class_name, "CHUMPA"),
                        "valid": False,
                        "structural_class": class_name,
                        "state": JACKET_STATE_BY_CLASS.get(class_name),
                        "priority": 2 if JACKET_STATE_BY_CLASS.get(class_name) == "close" else 1,
                    })
                elif logical_name == 'shirt':
                    detected["shirt"].append({
                        "crop": crop, "box": [x1, y1, x2, y2],
                        "display_name": "CAMISA", "valid": False,
                        "structural_class": class_name, "priority": 0,
                    })
                elif logical_name == 'pants':
                    has_pants_structure = True
                    detected["pants"].append({
                        "crop": crop, "box": [x1, y1, x2, y2],
                        "display_name": "PANTALON", "valid": False,
                        "structural_class": class_name, "priority": 0,
                    })
                elif logical_name == 'accesory':
                    detected["accesory"].append({
                        "crop": crop, "box": [x1, y1, x2, y2],
                        "display_name": "ACCESORIO", "valid": False,
                        "structural_class": class_name, "priority": 0,
                    })

        for key in detected:
            detected[key] = ClothingEngine._nms_clothing(detected[key])

        # --- SHORT-CIRCUIT: Accesorio ---
        # Si encuentra un accesorio, se detiene la validación DINOv2 por completo
        # y solo resalta el accesorio para ahorrar tiempo de cómputo.
        if len(detected["accesory"]) > 0:
            clothing_boxes = []
            for item in detected["accesory"]:
                clothing_boxes.append({
                    "class": item["display_name"],
                    "box":   item["box"],
                    "valid": False,
                })
            return (False, "ACCESORIO NO PERMITIDO", not has_pants_structure, clothing_boxes)

        details_log      = []
        has_top_valid    = False
        has_bottom_valid = False
        jacket_estado    = None

        for item in detected["jacket"]:
            is_valid, meta, debug = ClothingEngine._validate_garment_embedding(
                item["crop"], "jacket"
            )
            print(f"  [VALIDATE][JACKET] {debug}")
            if is_valid:
                item["valid"] = True
                inferred_state = ClothingEngine._infer_jacket_state(
                    item.get("structural_class"),
                    meta,
                )
                item["state"] = inferred_state
                if inferred_state == "close":
                    jacket_estado = "close"
                    details_log.append("Chumpa CERRADA oficial OK")
                    break
                if inferred_state == "open" and jacket_estado != "close":
                    jacket_estado = "open"
                    details_log.append("Chumpa ABIERTA oficial OK")
                    continue
                jacket_estado = jacket_estado or "close"
                details_log.append("Chumpa oficial OK")
                break

        has_valid_shirt = False
        for item in detected["shirt"]:
            is_valid, meta, debug = ClothingEngine._validate_garment_embedding(
                item["crop"], "shirt"
            )
            print(f"  [VALIDATE][SHIRT] {debug}")
            if is_valid:
                item["valid"]    = True
                has_valid_shirt  = True
                details_log.append("Camisa oficial OK")
                break

        if jacket_estado == 'close':
            has_top_valid = True
        elif jacket_estado == 'open':
            if has_valid_shirt:
                has_top_valid = True
                details_log.append("Chumpa abierta + camisa oficial OK")
            else:
                has_top_valid = False
                details_log.append("Chumpa abierta requiere camisa oficial")
        elif has_valid_shirt:
            has_top_valid = True
        else:
            if detected["jacket"] or detected["shirt"]:
                details_log.append("Prenda Superior No de Catálogo")

        for item in detected["pants"]:
            is_valid, meta, debug = ClothingEngine._validate_garment_embedding(
                item["crop"], "pants"
            )
            print(f"  [VALIDATE][PANTS] {debug}")
            if is_valid:
                item["valid"]    = True
                has_bottom_valid = True
                details_log.append("Pantalón oficial OK")
                break

        if not has_bottom_valid:
            if detected["pants"]:
                details_log.append("Pantalón No de Catálogo")
            else:
                details_log.append("Sin Pantalón Visible")

        if not details_log:
            details_log.append("Ropa no detectada por YOLO")

        clothing_boxes = []
        # Excluir accesorios aquí porque ya fueron filtrados en el short-circuit superior
        for cat_name, cat in detected.items():
            if cat_name == "accesory":
                continue
            for item in cat:
                clothing_boxes.append({
                    "class": item["display_name"],
                    "box":   item["box"],
                    "valid": item["valid"],
                })

        uniform_ok = has_top_valid and has_bottom_valid
        return (
            uniform_ok,
            " | ".join(details_log),
            not has_pants_structure,
            clothing_boxes,
        )
