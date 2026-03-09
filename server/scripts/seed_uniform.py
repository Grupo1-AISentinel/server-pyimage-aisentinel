"""
seed_uniform.py — Registro multi-clase de uniformes en ChromaDB.

ESTRUCTURA DE DIRECTORIOS ESPERADA:
  img/uniforms/
    <marca>/
      close/   → chumpas cerradas (ej. clasic/close/, promo33/close/)
      open/    → chumpas abiertas (ej. clasic/open/, promo33/open/)
    tshirt/    → camisas oficiales (sin subdirectorios open/close)

ESTRATEGIA DE VECTORES INDIVIDUALES:
  En lugar de promediar todos los vectores en UNO por prenda, se guarda CADA
  vector detectado como una entrada separada en ChromaDB.

  Por qué es mejor:
    - El promedio "difumina" las diferencias intra-clase → peor discriminación.
    - Con vectores individuales, ChromaDB busca el vecino más cercano entre
      todos los ejemplares registrados → encuentra el ejemplo más similar.
    - Una chumpa con iluminación particular matchea con el ejemplar registrado
      en iluminación similar, no con el centroide difuso de todas.

  IDs: <base_id>_<índice>  (ej. "clasic_jacket_close_0", "camisa_oficial_12")
  Todos los vectores del mismo tipo llevan el mismo metadata["tipo"] y
  metadata["base_id"] para trazar su origen.

BÚSQUEDA TIPADA (crud_uniform.py):
  search_uniform_by_vector(vector, tipo="jacket") busca solo entre
  los vectores cuyo metadata["tipo"] == "jacket".
  → Elimina la confusión cross-tipo que causaba falsos rechazos.

ESTADOS DE JACKET:
  - close: chumpa cerrada → la camisa NO es obligatoria en validación
  - open:  chumpa abierta → la camisa SÍ es obligatoria en validación
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


def extract_vectors_from_images(images: list[str], priority_class: str) -> tuple:
    """
    Procesa cada imagen con YOLO y acumula vectores 512D + histogramas HS.

    Retorna:
        (vector_pools, color_pools)
        vector_pools: {"jacket": [v0, ...], "shirt": [...], "pant": [...]}
        color_pools:  {"jacket": [hist0, ...], "shirt": [...], "pant": [...]}
    """
    pools: dict[str, list] = {"jacket": [], "shirt": [], "pant": []}
    color_pools: dict[str, list] = {"jacket": [], "shirt": [], "pant": []}
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
                hs_hist = ClothingEngine._compute_hs_histogram(crop)
                if hs_hist is not None:
                    color_pools[yolo_class].append(hs_hist)

        detected_summary = {k: len(v) for k, v in detections.items() if v}
        print(f"    {os.path.basename(img_path)}: {detected_summary}")

    print(f"  → {priority_found} detecciones de '{priority_class}' (prioritaria) en {len(images)} imágenes")
    return pools, color_pools


def save_pool_individual(item_id_prefix: str, tipo: str, vectors: list,
                         extra_meta: dict = None) -> int:
    """
    Guarda CADA vector como una entrada independiente en ChromaDB.

    ID: <item_id_prefix>_<índice>  (ej. "clasic_jacket_close_0")
    metadata incluye siempre:
      - tipo:    categoria de prenda ("jacket", "shirt", "pants")
      - valido:  True
      - base_id: prefijo sin índice (para agrupar variantes del mismo item)
      - (+ los campos de extra_meta: estado, marca, etc.)

    Retorna el número de vectores guardados.
    """
    if not vectors:
        print(f"  [SKIP] {item_id_prefix}: sin vectores para guardar.")
        return 0

    for i, vector in enumerate(vectors):
        metadata = {"tipo": tipo, "valido": True, "base_id": item_id_prefix}
        if extra_meta:
            metadata.update(extra_meta)
        upsert_uniform_vector(f"{item_id_prefix}_{i}", vector, metadata)

    print(f"  ✅ '{item_id_prefix}' ({tipo}) — {len(vectors)} vectores individuales guardados.")
    return len(vectors)


def populate_uniforms():
    base_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    uniforms_dir = os.path.join(base_dir, "img", "uniforms")

    print("=" * 65)
    print("  SEED UNIFORMES — Embeddings augmentados (ResNet+Color 728D)")
    print("=" * 65)

    from db.connection import client
    try:
        client.delete_collection("uniform_catalog")
        print("[RESET] Colección 'uniform_catalog' eliminada (nueva dimensión 728D).")
    except Exception:
        print("[RESET] Colección 'uniform_catalog' no existía, se creará nueva.")

    global_shirt_vectors = []
    global_pants_vectors = []
    global_jacket_colors = []
    global_shirt_colors  = []
    global_pants_colors  = []

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
            pools, color_pools = extract_vectors_from_images(images, priority_class="jacket")

            save_pool_individual(
                f"{brand}_jacket_{estado}",
                "jacket",
                pools["jacket"],
                extra_meta={"estado": estado, "marca": brand},
            )

            global_shirt_vectors.extend(pools["shirt"])
            global_pants_vectors.extend(pools["pant"])
            global_jacket_colors.extend(color_pools["jacket"])
            global_shirt_colors.extend(color_pools["shirt"])
            global_pants_colors.extend(color_pools["pant"])
            if pools["shirt"]:
                print(f"  + {len(pools['shirt'])} vectores de camisa al pool global.")
            if pools["pant"]:
                print(f"  + {len(pools['pant'])} vectores de pantalón al pool global.")

    # --- 2. Procesar camisas dedicadas (tshirt/) ---
    tshirt_dir    = os.path.join(uniforms_dir, "tshirt")
    tshirt_images = get_images(tshirt_dir)

    if tshirt_images:
        print(f"\n--- TSHIRT ({len(tshirt_images)} imágenes) ---")
        pools, color_pools = extract_vectors_from_images(tshirt_images, priority_class="shirt")
        global_shirt_vectors.extend(pools["shirt"])
        global_pants_vectors.extend(pools["pant"])
        global_shirt_colors.extend(color_pools["shirt"])
        global_pants_colors.extend(color_pools["pant"])
        if pools["shirt"]:
            print(f"  + {len(pools['shirt'])} vectores de camisa al pool global.")
    else:
        print("\n[SKIP] tshirt/: sin imágenes.")

    # --- 3. Guardar pools globales (vectores individuales) ---
    print("\n--- GUARDANDO POOLS GLOBALES ---")
    total_shirts = save_pool_individual("camisa_oficial",   "shirt", global_shirt_vectors)
    total_pants  = save_pool_individual("pantalon_oficial", "pants", global_pants_vectors)

    # --- 4. Guardar referencias de color (histogramas HS medianos) ---
    print("\n--- GUARDANDO REFERENCIAS DE COLOR ---")
    ClothingEngine.save_color_reference("jacket", global_jacket_colors)
    ClothingEngine.save_color_reference("shirt",  global_shirt_colors)
    ClothingEngine.save_color_reference("pants",  global_pants_colors)

    print(f"\n  Camisas en BD:   {total_shirts} vectores")
    print(f"  Pantalones en BD: {total_pants} vectores")
    print(f"  Refs de color:    jacket={len(global_jacket_colors)} shirt={len(global_shirt_colors)} pants={len(global_pants_colors)}")
    print("\n" + "=" * 65)
    print("  COMPLETADO: uniforme registrado en ChromaDB + refs de color.")
    print("=" * 65)


if __name__ == "__main__":
    populate_uniforms()
