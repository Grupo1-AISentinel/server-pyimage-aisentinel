"""
seed_uniform.py — Registro multi-clase de uniformes en ChromaDB.
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.clothing_engine import ClothingEngine
from db.cruds.crud_uniform import upsert_uniform_vector

VALID_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

# Confianza baja en el seeder: queremos capturar TODAS las prendas visibles,
# incluso parcialmente ocluidas o con iluminación difícil.
# En inferencia en vivo se usa conf=0.30 (clothing_engine.py).
SEEDER_CONF = 0.20


def get_images(dir_path: str) -> list[str]:
    if not os.path.isdir(dir_path):
        return []
    return sorted([
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if os.path.splitext(f.lower())[1] in VALID_EXT
    ])


def extract_vectors_from_images(images: list[str], priority_class: str) -> dict:
    """
    Procesa cada imagen con YOLO y acumula embeddings de DINOv2.

    Retorna:
        pools: {"jacket": [v0, ...], "shirt": [...], "pants": [...]}
    """
    pools: dict[str, list] = {"jacket": [], "shirt": [], "pants": []}
    priority_found = 0

    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [WARN] No se pudo leer: {os.path.basename(img_path)}")
            continue

        detections = ClothingEngine.extract_all_clothing_from_image(
            frame, conf=SEEDER_CONF
        )

        for yolo_class, crops in detections.items():
            for crop in crops:
                vector = ClothingEngine._get_embedding(crop)
                if vector:
                    pools[yolo_class].append(vector)
                    if yolo_class == priority_class:
                        priority_found += 1

        detected_summary = {k: len(v) for k, v in detections.items() if v}
        print(f"    {os.path.basename(img_path)}: {detected_summary}")

    print(f"  → {priority_found} detecciones de '{priority_class}' (prioritaria) en {len(images)} imágenes")
    return pools


def save_pool_individual(item_id_prefix: str, tipo: str, vectors: list,
                         extra_meta: dict = None) -> int:
    """
    
    """
    if not vectors:
        print(f"  [SKIP] {item_id_prefix}: sin vectores para guardar.")
        return 0

    for i, vector in enumerate(vectors):
        metadata = ClothingEngine._build_uniform_metadata(
            {
                "tipo": tipo,
                "base_id": item_id_prefix,
                **(extra_meta or {}),
            },
            source="seed_uniform",
        )
        upsert_uniform_vector(f"{item_id_prefix}_{i}", vector, metadata)

    print(f"  ✅ '{item_id_prefix}' ({tipo}) — {len(vectors)} vectores individuales guardados.")
    return len(vectors)


def populate_uniforms():
    base_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    uniforms_dir = os.path.join(base_dir, "img", "uniforms")

    print("=" * 65)
    print("  SEED UNIFORMES — Embeddings de DINOv2 (384D)")
    print("=" * 65)

    from db.connection import UNIFORMS_COLLECTION_NAME, client
    reset_collection = os.getenv("RESET_UNIFORM_COLLECTION", "true").lower() == "true"
    
    if reset_collection:
        print(f"[INFO] Eliminando colección '{UNIFORMS_COLLECTION_NAME}' para purgar ropas antiguas...")
        try:
            client.delete_collection(name=UNIFORMS_COLLECTION_NAME)
            print("[INFO] Colección eliminada exitosamente. Se creará una nueva limpia.")
        except Exception:
            pass

    global_shirt_vectors = []
    global_pants_vectors = []

    # --- 1. Procesar marcas de chumpas (subdirectorios con close/ y open/) ---
    try:
        brands = sorted([
            d for d in os.listdir(uniforms_dir)
            if os.path.isdir(os.path.join(uniforms_dir, d)) and d != "tshirt"
        ])
    except FileNotFoundError:
        print(f"[ERROR] Directorio no encontrado: {uniforms_dir}")
        return

    for brand in brands:
        brand_dir = os.path.join(uniforms_dir, brand)

        for estado in ("close", "open"):
            estado_dir = os.path.join(brand_dir, estado)
            images     = get_images(estado_dir)

            if not images:
                print(f"\n[SKIP] {brand}/{estado}: sin imágenes.")
                continue

            print(f"\n--- {brand.upper()} / {estado.upper()} ({len(images)} imágenes) ---")
            pools = extract_vectors_from_images(images, priority_class="jacket")

            save_pool_individual(
                f"{brand}_jacket_{estado}",
                "jacket",
                pools["jacket"],
                extra_meta={"estado": estado, "marca": brand},
            )

            global_shirt_vectors.extend(pools["shirt"])
            global_pants_vectors.extend(pools["pants"])
            if pools["shirt"]:
                print(f"  + {len(pools['shirt'])} vectores de camisa al pool global.")
            if pools["pants"]:
                print(f"  + {len(pools['pants'])} vectores de pantalón al pool global.")

    # --- 2. Procesar camisas dedicadas (tshirt/) ---
    tshirt_dir    = os.path.join(uniforms_dir, "tshirt")
    tshirt_images = get_images(tshirt_dir)

    if tshirt_images:
        print(f"\n--- TSHIRT ({len(tshirt_images)} imágenes) ---")
        pools = extract_vectors_from_images(tshirt_images, priority_class="shirt")
        global_shirt_vectors.extend(pools["shirt"])
        global_pants_vectors.extend(pools["pants"])
        if pools["shirt"]:
            print(f"  + {len(pools['shirt'])} vectores de camisa al pool global.")
    else:
        print("\n[SKIP] tshirt/: sin imágenes.")

    # --- 3. Guardar pools globales (vectores individuales) ---
    print("\n--- GUARDANDO POOLS GLOBALES ---")
    total_shirts = save_pool_individual("camisa_oficial",   "shirt", global_shirt_vectors)
    total_pants  = save_pool_individual("pantalon_oficial", "pants", global_pants_vectors)

    print(f"\n  Camisas en BD:   {total_shirts} vectores")
    print(f"  Pantalones en BD: {total_pants} vectores")
    print("\n" + "=" * 65)
    print("  COMPLETADO: uniforme registrado en ChromaDB.")
    print("=" * 65)


if __name__ == "__main__":
    populate_uniforms()
