"""
totem_testimages.py — Totem de prueba sobre imágenes estáticas.

Lee imágenes desde:   assets/test-images/
Guarda resultados en: assets/valid-images/

Envía cada imagen al servidor AI Sentinel vía HTTP POST (/api/detect) y dibuja
sobre la imagen original todos los resultados (caras, identidades, ropa, uniforme).

Uso:
  cd clients/totem
  python totem_testimages.py

Requiere que el servidor esté corriendo en URL_SERVIDOR.
"""
import os
import sys
import cv2
import requests
import numpy as np

# Agregar ruta para que ui_utils.py pueda ser importado
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ui_utils import draw_futuristic_box

URL_SERVIDOR = "http://localhost:8000"
VALID_EXT    = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

# Colores
COLOR_OK         = (0,  200,   0)   # verde
COLOR_NOK        = (0,    0, 220)   # rojo
COLOR_DESCONOCIDO = (0,  165, 255)  # naranja
COLOR_ROPA_OK    = (0,  200,   0)
COLOR_ROPA_NOK   = (0,    0, 200)


def _detectar(img_path: str) -> dict | None:
    """POST de la imagen al servidor, retorna JSON o None si falla."""
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"  [WARN] No se pudo leer: {os.path.basename(img_path)}")
        return None

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    try:
        resp = requests.post(
            f"{URL_SERVIDOR}/api/detect",
            files={"file": (os.path.basename(img_path), buf.tobytes(), "image/jpeg")},
            timeout=60,
        )
        resp.raise_for_status()
        return frame, resp.json()
    except requests.exceptions.ConnectionError:
        print(f"  [ERROR] No se puede conectar a {URL_SERVIDOR} — ¿está el servidor corriendo?")
        return None, None
    except Exception as e:
        print(f"  [ERROR] {os.path.basename(img_path)}: {e}")
        return frame, None


def _dibujar_resultados(frame: np.ndarray, students: list) -> np.ndarray:
    """Dibuja bounding boxes, etiquetas e información de uniforme sobre el frame."""
    annotated = frame.copy()

    for student in students:
        top, right, bottom, left = student.get("location", [0, 0, 0, 0])
        identity   = student.get("identity",   "Desconocido")
        has_unif   = student.get("has_uniform", False)
        confidence = student.get("confidence", "")
        clothing_d = student.get("clothing_details", "")
        needs_body = student.get("needs_full_body_view", False)

        # Color del recuadro
        if identity == "Desconocido":
            color = COLOR_DESCONOCIDO
        elif has_unif:
            color = COLOR_OK
        else:
            color = COLOR_NOK

        # Recuadro futurista de cara
        unif_str = "UNIFORME OK" if has_unif else "SIN UNIFORME"
        conf_str = f" ({confidence})" if confidence else ""
        label    = f"{identity}{conf_str} | {unif_str}"
        
        draw_futuristic_box(annotated, left, top, right, bottom, color, text=label, text_size=0.6)

        # Detalles de ropa (clothing_details) debajo del recuadro
        if clothing_d:
            cv2.putText(annotated, clothing_d,
                        (left, min(bottom + 20, annotated.shape[0] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        # Bounding boxes de prendas individuales
        for cb in student.get("clothing_boxes", []):
            bx1, by1, bx2, by2 = cb["box"]
            c_color = COLOR_ROPA_OK if cb.get("valid") else COLOR_ROPA_NOK
            draw_futuristic_box(annotated, bx1, by1, bx2, by2, c_color, text=cb["class"].upper(), text_size=0.5)

        # Aviso de cuerpo incompleto
        if needs_body:
            h = annotated.shape[0]
            cv2.putText(annotated, "ACERQUESE/ALEJESE PARA VER PANTALON",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return annotated


def process_test_images():
    base       = os.path.dirname(os.path.abspath(__file__))
    input_dir  = os.path.normpath(os.path.join(base, "..", "assets", "test-images"))
    output_dir = os.path.normpath(os.path.join(base, "..", "assets", "valid-images"))
    os.makedirs(output_dir, exist_ok=True)

    # Colectar imágenes
    images = sorted([
        f for f in os.listdir(input_dir)
        if os.path.splitext(f.lower())[1] in VALID_EXT
    ])

    if not images:
        print(f"[TestImages] No hay imágenes en {input_dir}")
        return

    print("=" * 65)
    print(f"  AI Sentinel — Detección en imágenes estáticas")
    print(f"  Entrada : {input_dir}")
    print(f"  Salida  : {output_dir}")
    print(f"  Imágenes: {len(images)}")
    print("=" * 65)

    ok_count  = 0
    err_count = 0

    for fname in images:
        img_path  = os.path.join(input_dir, fname)
        out_path  = os.path.join(output_dir, fname)

        print(f"\n[{fname}]")
        frame, data = _detectar(img_path)

        if frame is None:
            err_count += 1
            continue

        if data is None:
            # Guardar imagen sin anotar si el servidor falló
            cv2.imwrite(out_path, frame)
            err_count += 1
            continue

        students = data.get("students", [])
        print(f"  {len(students)} persona(s) detectada(s):")

        for s in students:
            ident   = s.get("identity", "Desconocido")
            unif    = "✅ Uniforme" if s.get("has_uniform") else "❌ Sin uniforme"
            details = s.get("clothing_details", "")
            print(f"    → {ident}: {unif} | {details}")

        annotated = _dibujar_resultados(frame, students)
        cv2.imwrite(out_path, annotated)
        print(f"  Guardado → {out_path}")
        ok_count += 1

    print("\n" + "=" * 65)
    print(f"  COMPLETADO: {ok_count} procesadas, {err_count} con error.")
    print("=" * 65)


if __name__ == "__main__":
    process_test_images()
