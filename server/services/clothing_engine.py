import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from ultralytics import YOLO
import cv2
import numpy as np

from db.cruds.crud_uniform import (
    save_uniform_vector, search_uniform_by_vector,
    upsert_uniform_vector, search_uniform_by_vector_topk,
)

# ---------------------------------------------------------------------------
# Device flags
# DEVICE_YOLO:  índice GPU para Ultralytics ('0') o 'cpu'.
# DEVICE_TORCH: string para PyTorch .to() → 'cuda' o 'cpu'.
# USE_HALF:     FP16 en inferencia YOLO → ~25% speedup en memoria bandwidth.
# ---------------------------------------------------------------------------
DEVICE_YOLO  = '0'    if torch.cuda.is_available() else 'cpu'
DEVICE_TORCH = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_HALF     = torch.cuda.is_available()

if torch.cuda.is_available():
    _gpu_name = torch.cuda.get_device_name(0)
    _vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"[CUDA][ClothingEngine] ✅ GPU: {_gpu_name} | VRAM: {_vram_gb:.1f} GB | FP16={USE_HALF}")
else:
    print("[CUDA][ClothingEngine] ⚠️  Sin GPU — YOLO y ResNet en CPU")

# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
print("[INFO][ClothingEngine] Cargando modelos YOLO persona y ropa...")
yolo_person_model            = YOLO('yolov8n.pt')
yolo_clothing_structure_model = YOLO('best.pt')

# ResNet18 como extractor de vectores de ropa (512D)
resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
resnet_model.eval()
resnet_model = resnet_model.to(DEVICE_TORCH)

# torch.compile (PyTorch ≥ 2.0) requiere CUDA Capability ≥ 7.0 (Volta+)
# porque usa Triton como backend de compilación. Pascal (GTX 1050 Ti = cc 6.1)
# no lo soporta → lo omitimos y seguimos en eager mode (sin pérdida de funcionalidad).
_cuda_cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
if _cuda_cc[0] >= 7:
    try:
        resnet_model = torch.compile(resnet_model)
        print("[INFO][ClothingEngine] ✅ torch.compile aplicado a ResNet18.")
    except Exception as e:
        print(f"[INFO][ClothingEngine] torch.compile falló ({e}), usando eager mode.")
else:
    print(f"[INFO][ClothingEngine] torch.compile omitido (cc {_cuda_cc[0]}.{_cuda_cc[1]} < 7.0, requiere Volta+). Eager mode OK.")

# Pipeline de transformación preconstruido (se crea UNA sola vez)
RESNET_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Warmup de modelos YOLO
# La primera inferencia en CUDA siempre es lenta por compilación de kernels.
# Absorber ese costo en startup evita que el primer frame real sea lento.
# ---------------------------------------------------------------------------
print("[INFO][ClothingEngine] Warming up modelos YOLO en GPU...")
_dummy_person   = np.zeros((320, 180, 3), dtype=np.uint8)
_dummy_clothing = np.zeros((200, 100, 3), dtype=np.uint8)
yolo_person_model(_dummy_person,    classes=[0], device=DEVICE_YOLO, verbose=False, imgsz=320, half=USE_HALF, conf=0.30)
yolo_clothing_structure_model(_dummy_clothing,   device=DEVICE_YOLO, verbose=False, imgsz=320, half=USE_HALF, conf=0.30)

# Warmup de ResNet
with torch.no_grad():
    _dummy_tensor = torch.zeros(1, 3, 224, 224).to(DEVICE_TORCH)
    resnet_model(_dummy_tensor)

print("[INFO][ClothingEngine] ✅ Modelos listos.")

# ---------------------------------------------------------------------------
# Color analysis: HS histograms & multi-signal validation thresholds
# ---------------------------------------------------------------------------
COLOR_REF_DIR = os.getenv("COLOR_REF_DIR", "./color_references")
os.makedirs(COLOR_REF_DIR, exist_ok=True)

BINS_H = 30   # Hue buckets   (0-180 in OpenCV HSV)
BINS_S = 32   # Saturation buckets (0-256)

# ChromaDB cosine distance thresholds (1 - cosine_similarity).
# Compared against MEDIAN of top-k results for robustness.
DIST_THRESHOLD = {
    "jacket": 0.50,
    "shirt":  0.50,
    "pants":  0.45,
}

# Minimum HS histogram correlation vs stored reference (-1..1, higher=better)
COLOR_CORR_THRESHOLD = {
    "jacket": 0.20,
    "shirt":  0.20,
    "pants":  0.30,
}

# Maximum Bhattacharyya distance vs stored reference (0..1, lower=better)
BHATTA_THRESHOLD = {
    "jacket": 0.75,
    "shirt":  0.75,
    "pants":  0.60,
}

TOP_K_RESULTS = 5

# ---------------------------------------------------------------------------
# Augmented embedding: ResNet (shape) + HS histogram (color) in one vector.
# ChromaDB searches inherently consider BOTH shape AND color, eliminating
# false positives from structurally similar but differently colored garments.
# ---------------------------------------------------------------------------
EMB_BINS_H     = 18      # Hue bins for embedding (each covers 10 degrees)
EMB_BINS_S     = 12      # Saturation bins for embedding
WEIGHT_RESNET  = 0.35    # Shape/texture contribution
WEIGHT_COLOR   = 0.65    # Color distribution contribution (dominant)


class ClothingEngine:

    @staticmethod
    def _iou(box_a, box_b) -> float:
        """Intersection over Union entre dos cajas [x1,y1,x2,y2]."""
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
        """
        Non-Maximum Suppression entre prendas del mismo tipo.

        Problema: YOLO detecta tanto 'jacket' como 'shirt' en el mismo torso
        (o el mismo tipo dos veces) con cajas muy solapadas. Sin NMS, el render
        dibuja dos rectángulos sobre la misma prenda → "doble chumpa".

        Estrategia: ordenar por área descendente (el bounding-box más grande
        suele ser la detección más completa), descartar cualquier caja posterior
        cuyo IoU con una ya aceptada supere iou_thresh.
        """
        if len(items) <= 1:
            return items

        # Ordenar por área descendente
        sorted_items = sorted(
            items,
            key=lambda it: (it["box"][2] - it["box"][0]) * (it["box"][3] - it["box"][1]),
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
        """
        Detecta personas en el frame y devuelve sus recortes.
        FP16 activo en GPU. imgsz=320 apropiado para frame 180x320.
        """
        results = yolo_person_model(
            frame, classes=[0], device=DEVICE_YOLO,
            verbose=False, imgsz=320, half=USE_HALF
        )
        personas = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size != 0:
                    personas.append({"box": (x1, y1, x2, y2), "crop": crop})
        return personas

    @staticmethod
    def _get_embedding(cropped_image_bgr):
        """
        Color-augmented embedding: ResNet18 512D (shape/texture) concatenated
        with HS histogram 216D (color), weighted so color dominates.
        Total: 728D, L2-normalized for cosine similarity in ChromaDB.

        Why this works: ResNet alone captures "it's pants" but NOT "it's grey
        pants vs black pants". By fusing color distribution directly into the
        embedding vector, ChromaDB nearest-neighbor search inherently prefers
        garments that match in BOTH shape AND color.
        """
        if cropped_image_bgr is None or cropped_image_bgr.size == 0:
            return None

        # --- ResNet semantic features (512D) ---
        color_rgb  = cv2.cvtColor(cropped_image_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = RESNET_TRANSFORM(color_rgb).unsqueeze(0).to(DEVICE_TORCH)
        with torch.no_grad():
            resnet_vec = resnet_model(img_tensor).flatten()
            resnet_vec = torch.nn.functional.normalize(resnet_vec, p=2, dim=0)
            resnet_np  = resnet_vec.cpu().numpy()

        # --- HS color histogram (EMB_BINS_H * EMB_BINS_S = 216D) ---
        hsv  = cv2.cvtColor(cropped_image_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 25, 35]), np.array([180, 255, 225]))
        if cv2.countNonZero(mask) < 50:
            mask = None
        hist = cv2.calcHist([hsv], [0, 1], mask,
                            [EMB_BINS_H, EMB_BINS_S],
                            [0, 180, 0, 256])
        hist_flat = hist.flatten().astype(np.float32)
        h_norm = np.linalg.norm(hist_flat)
        if h_norm > 0:
            hist_flat /= h_norm

        # Weighted concatenation: color dominates for better discrimination
        composite = np.concatenate([
            resnet_np * WEIGHT_RESNET,
            hist_flat * WEIGHT_COLOR,
        ])
        c_norm = np.linalg.norm(composite)
        if c_norm > 0:
            composite /= c_norm

        return composite.tolist()

    @staticmethod
    def extract_all_clothing_from_image(frame_bgr, conf: float = 0.20) -> dict:
        """
        Pasa el frame completo por YOLO y devuelve todos los crops detectados
        agrupados por clase: {"jacket": [...], "shirt": [...], "pant": [...]}.

        Parámetro conf:
          Umbral de confianza de detección. El seeder usa 0.20 (captura todas las
          prendas visibles aunque sea con baja confianza → más vectores para la BD).
          En inferencia en vivo se usa 0.30 (menos ruido, solo detecciones seguras).

        imgsz=640: máxima resolución nativa del modelo → el seeder corre sin
        restricciones de tiempo, así que aprovechamos toda la precisión disponible.
        """
        results = yolo_clothing_structure_model(
            frame_bgr, device=DEVICE_YOLO, verbose=False,
            imgsz=640, half=USE_HALF, conf=conf
        )
        detected: dict[str, list] = {"jacket": [], "shirt": [], "pant": []}
        for result in results:
            for box in result.boxes:
                class_name = result.names[int(box.cls[0])]
                if class_name in detected:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame_bgr[y1:y2, x1:x2]
                    if crop.size > 0:
                        detected[class_name].append(crop)
        return detected

    @staticmethod
    def register_clothing_item(item_id: str, item_type: str, images_list: list,
                               extra_meta: dict = None):
        """
        Registra una prenda de uniforme en ChromaDB.
        extra_meta permite añadir campos adicionales como 'estado': 'open'/'close'.
        """
        yolo_class = {"jacket": "jacket", "shirt": "shirt",
                      "pants": "pant", "accesory": "accesory"}.get(item_type)
        if not yolo_class:
            return False

        vectors_buffer = []
        for image_path in images_list:
            frame = cv2.imread(image_path)
            if frame is None:
                continue
            results = yolo_clothing_structure_model(
                frame, device=DEVICE_YOLO, verbose=False, imgsz=640, half=USE_HALF
            )
            for result in results:
                for box in result.boxes:
                    if result.names[int(box.cls[0])] == yolo_class:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            v = ClothingEngine._get_embedding(crop)
                            if v:
                                vectors_buffer.append(v)

        if not vectors_buffer:
            print(f"[ERROR] YOLO no detectó '{item_type}' en las imágenes de {item_id}.")
            return False

        master_vector = np.mean(vectors_buffer, axis=0).tolist()
        metadata = {"tipo": item_type, "valido": True}
        if extra_meta:
            metadata.update(extra_meta)
        upsert_uniform_vector(item_id, master_vector, metadata)
        return True

    # ------------------------------------------------------------------
    # Color analysis helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hs_histogram(crop_bgr):
        """
        2D Hue-Saturation histogram in HSV space.
        V (brightness) is ignored → invariant to lighting changes.
        Very dark / very bright / low-saturation pixels are masked out because
        their hue is unreliable.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 30, 40]), np.array([180, 255, 220]))
        if cv2.countNonZero(mask) < 100:
            mask = None
        hist = cv2.calcHist([hsv], [0, 1], mask,
                            [BINS_H, BINS_S], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.astype(np.float32)

    @staticmethod
    def _compare_color(hist_a, hist_b):
        """Correlation + Bhattacharyya between two HS histograms."""
        if hist_a is None or hist_b is None:
            return (0.0, 1.0)
        corr   = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
        bhatta = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA)
        return (corr, bhatta)

    @staticmethod
    def _get_color_ref_path(tipo: str) -> str:
        return os.path.join(COLOR_REF_DIR, f"{tipo}_hs_ref.npy")

    _color_refs: dict = {}

    @staticmethod
    def load_color_reference(tipo: str):
        """Load cached HS reference histogram for a garment type."""
        if tipo in ClothingEngine._color_refs:
            return ClothingEngine._color_refs[tipo]
        ref_path = ClothingEngine._get_color_ref_path(tipo)
        if os.path.exists(ref_path):
            ref = np.load(ref_path).astype(np.float32)
            ClothingEngine._color_refs[tipo] = ref
            return ref
        return None

    @staticmethod
    def save_color_reference(tipo: str, histograms: list):
        """Compute and persist the median HS histogram as reference."""
        if not histograms:
            return
        stacked     = np.stack(histograms, axis=0)
        median_hist = np.median(stacked, axis=0).astype(np.float32)
        cv2.normalize(median_hist, median_hist, 0, 1, cv2.NORM_MINMAX)
        ref_path    = ClothingEngine._get_color_ref_path(tipo)
        np.save(ref_path, median_hist)
        ClothingEngine._color_refs.pop(tipo, None)
        print(f"  [COLOR-REF] '{tipo}' guardada ({len(histograms)} muestras)")

    # ------------------------------------------------------------------
    # Multi-signal garment validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_garment_multisignal(crop, tipo: str):
        """
        Two-gate validation — both must pass:
          Gate 1: ResNet embedding top-k consensus (median cosine distance)
          Gate 2: HS color histogram correlation + Bhattacharyya vs reference
        Falls back to a much stricter embedding-only check when no color ref
        is available.
        """
        vector = ClothingEngine._get_embedding(crop)
        if vector is None:
            return False, None, "embedding=None"

        topk = search_uniform_by_vector_topk(vector, tipo=tipo, k=TOP_K_RESULTS)
        if topk is None:
            return False, None, f"sin_ref({tipo})"

        median_dist = topk["median_distance"]
        best_dist   = topk["top_distance"]
        dist_lim    = DIST_THRESHOLD.get(tipo, 0.45)
        embedding_ok = median_dist < dist_lim

        color_ref     = ClothingEngine.load_color_reference(tipo)
        detected_hist = ClothingEngine._compute_hs_histogram(crop)
        corr_lim      = COLOR_CORR_THRESHOLD.get(tipo, 0.20)
        bhatta_lim    = BHATTA_THRESHOLD.get(tipo, 0.70)

        if color_ref is not None and detected_hist is not None:
            corr, bhatta = ClothingEngine._compare_color(color_ref, detected_hist)
            color_ok = (corr > corr_lim) and (bhatta < bhatta_lim)
        else:
            corr, bhatta = -1.0, -1.0
            color_ok = embedding_ok and best_dist < (dist_lim * 0.65)

        is_valid = embedding_ok and color_ok

        debug = (
            f"best={best_dist:.3f} med={median_dist:.3f}(lim={dist_lim}) "
            f"corr={corr:.3f}(lim={corr_lim}) bhatta={bhatta:.3f}(lim={bhatta_lim}) "
            f"emb={'OK' if embedding_ok else 'X'} col={'OK' if color_ok else 'X'} "
            f"-> {'VALID' if is_valid else 'REJECT'}"
        )
        return is_valid, topk["metadata"], debug

    @staticmethod
    def validate_uniform(person_crop):
        """
        Pipeline multi-señal de validación de uniforme.

        Cada prenda pasa por DOS gates independientes:
          1. Embedding ResNet → top-k consensus (mediana de distancias)
          2. Histograma HS de color → correlación + Bhattacharyya vs referencia

        Reglas de negocio:
          - Jacket CLOSE → camisa NO obligatoria
          - Jacket OPEN  → camisa SÍ obligatoria
          - Solo camisa  → válido
          - Pantalón oficial SIEMPRE requerido
        """
        # imgsz=320: resolución óptima para crops de persona (el modelo fue entrenado
        # en ~640 pero los crops son subregiones pequeñas → 320 equilibra calidad/velocidad).
        # conf=0.30: umbral más alto que en el seeder (0.20) → solo detecciones seguras
        # para evitar falsos positivos en prendas de baja confianza.
        results = yolo_clothing_structure_model(
            person_crop, device=DEVICE_YOLO, verbose=False,
            imgsz=320, half=USE_HALF, conf=0.30
        )

        # Separar jacket, shirt, pants y accesory en listas independientes para NMS
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
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = person_crop[y1:y2, x1:x2]

                if class_name == 'jacket':
                    detected["jacket"].append({
                        "crop": crop, "box": [x1, y1, x2, y2],
                        "display_name": "CHUMPA", "valid": False,
                    })
                elif class_name == 'shirt':
                    detected["shirt"].append({
                        "crop": crop, "box": [x1, y1, x2, y2],
                        "display_name": "CAMISA", "valid": False,
                    })
                elif class_name == 'pant':
                    has_pants_structure = True
                    detected["pants"].append({
                        "crop": crop, "box": [x1, y1, x2, y2],
                        "display_name": "PANTALON", "valid": False,
                    })
                elif class_name == 'accesory':
                    detected["accesory"].append({
                        "crop": crop, "box": [x1, y1, x2, y2],
                        "display_name": "ACCESORIO", "valid": False,
                    })

        # NMS por categoría (evita doble detección de la misma prenda)
        for key in detected:
            detected[key] = ClothingEngine._nms_clothing(detected[key])

        details_log      = []
        has_top_valid    = False
        has_bottom_valid = False
        jacket_estado    = None   # 'open' | 'close' | None

        # --- Validar jackets (multi-señal: embedding + color) ---
        for item in detected["jacket"]:
            is_valid, meta, debug = ClothingEngine._validate_garment_multisignal(
                item["crop"], "jacket"
            )
            print(f"  [VALIDATE][JACKET] {debug}")
            if is_valid:
                item["valid"]  = True
                jacket_estado  = meta.get('estado', 'close') if meta else 'close'
                details_log.append(f"Chumpa {jacket_estado.upper()} OK")
                break

        # --- Validar shirts (multi-señal) ---
        has_valid_shirt = False
        for item in detected["shirt"]:
            is_valid, meta, debug = ClothingEngine._validate_garment_multisignal(
                item["crop"], "shirt"
            )
            print(f"  [VALIDATE][SHIRT] {debug}")
            if is_valid:
                item["valid"]    = True
                has_valid_shirt  = True
                details_log.append("Camisa OK")
                break

        # --- Regla de negocio open/close ---
        if jacket_estado == 'close':
            has_top_valid = True
        elif jacket_estado == 'open':
            if has_valid_shirt:
                has_top_valid = True
                details_log.append("Chumpa abierta + camisa OK")
            else:
                has_top_valid = False
                details_log.append("Chumpa abierta requiere camisa oficial")
        elif has_valid_shirt:
            has_top_valid = True
        else:
            if detected["jacket"] or detected["shirt"]:
                details_log.append("Prenda Superior No-Oficial")

        # --- Validar pantalón (multi-señal, umbrales estrictos) ---
        for item in detected["pants"]:
            is_valid, meta, debug = ClothingEngine._validate_garment_multisignal(
                item["crop"], "pants"
            )
            print(f"  [VALIDATE][PANTS] {debug}")
            if is_valid:
                item["valid"]    = True
                has_bottom_valid = True
                details_log.append("Pantalón OK")
                break

        if not has_bottom_valid:
            if detected["pants"]:
                details_log.append("Pantalón No-Oficial")
            else:
                details_log.append("Sin Pantalón Visible")

        # --- Accesorios no permitidos (gorras, pañuelos, etc.) ---
        # Cualquier accesorio detectado invalida el uniforme.
        has_accesory = len(detected["accesory"]) > 0
        if has_accesory:
            details_log.append("ACCESORIO NO PERMITIDO")

        if not details_log:
            details_log.append("Ropa no detectada por YOLO")

        # Armar clothing_boxes para el frontend (todas las categorías, accesory siempre invalid=False)
        clothing_boxes = []
        for cat in detected.values():
            for item in cat:
                clothing_boxes.append({
                    "class": item["display_name"],
                    "box":   item["box"],
                    "valid": item["valid"],
                })

        uniform_ok = has_top_valid and has_bottom_valid and not has_accesory
        return (
            uniform_ok,
            " | ".join(details_log),
            not has_pants_structure,
            clothing_boxes,
        )
