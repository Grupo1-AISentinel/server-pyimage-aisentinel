"""
seed_students.py — Registro automático de estudiantes en ChromaDB.

Auto-descubre TODAS las imágenes en img/students/<directorio>/ sin necesidad
de listar manualmente cada archivo. Añadir una foto nueva al directorio
y volver a ejecutar el seeder actualiza el registro automáticamente.

Mapa directorio → (carnet, nombre):
  Si se agrega un nuevo estudiante, agregar su entrada en STUDENTS_MAP.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.biometric_engine import BiometricEngine

# Extensiones de imagen soportadas
VALID_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

# Mapa: nombre_directorio → (carnet, nombre_completo)
STUDENTS_MAP = {
    "itiguila":  ("2024295", "Isaac Tiguila"),
    "jyaxon":    ("2024442", "Jeferson Yaxon"),
    "asosa":     ("2024449", "Anderson Sosa"),
    "wflorian":  ("2024392", "Wilson Florian"),
    "alucero":   ("2021211", "Angel Lucero"),
}


def get_images_in_dir(dir_path: str) -> list[str]:
    """Retorna todas las imágenes válidas de un directorio, ordenadas."""
    return sorted([
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if os.path.splitext(f.lower())[1] in VALID_EXT
    ])


def populate_students():
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    students_dir = os.path.join(base_dir, "img", "students")

    print("=" * 60)
    print("  SEED ESTUDIANTES — Auto-discovery de imágenes")
    print("=" * 60)

    from db.connection import FACES_COLLECTION_NAME, client
    print(f"[INFO] Eliminando colección '{FACES_COLLECTION_NAME}' para evitar basura de pruebas anteriores...")
    try:
        client.delete_collection(name=FACES_COLLECTION_NAME)
        print("[INFO] Colección eliminada exitosamente.")
    except Exception:
        pass

    engine = BiometricEngine()

    # Descubrir automáticamente todos los subdirectorios
    try:
        all_dirs = sorted([
            d for d in os.listdir(students_dir)
            if os.path.isdir(os.path.join(students_dir, d))
        ])
    except FileNotFoundError:
        print(f"[ERROR] Directorio no encontrado: {students_dir}")
        return

    found_count = 0
    for dir_name in all_dirs:
        if dir_name not in STUDENTS_MAP:
            print(f"\n[WARN] Directorio '{dir_name}' sin mapeo en STUDENTS_MAP — omitido.")
            print(f"       Agregar: \"{dir_name}\": (\"CARNET\", \"Nombre Completo\")")
            continue

        carnet, nombre = STUDENTS_MAP[dir_name]
        img_dir = os.path.join(students_dir, dir_name)
        images  = get_images_in_dir(img_dir)

        print(f"\n--- {nombre} ({carnet}) | {len(images)} imágenes en '{dir_name}/' ---")

        if not images:
            print(f"  [WARN] Sin imágenes en {img_dir}")
            continue

        try:
            engine.register_face(carnet, nombre, images)
            found_count += 1
        except Exception as e:
            print(f"  [ERROR] {nombre}: {e}")

    print("\n" + "=" * 60)
    print(f"  COMPLETADO: {found_count}/{len(all_dirs)} estudiantes registrados.")
    print("=" * 60)


if __name__ == "__main__":
    populate_students()
