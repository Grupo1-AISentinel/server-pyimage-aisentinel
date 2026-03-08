import cv2
import socketio
import time
import threading

URL_SERVIDOR = "http://localhost:8000"
NOMBRE_VENTANA = "Totem AI Sentinel - En vivo"

# --- Resoluciones ---
SEND_W, SEND_H = 320, 240    # Frame enviado al servidor para IA (fijo, no cambiar)
DISPLAY_W = 640              # Se actualiza con la resolución real del video en open_cam()
DISPLAY_H = 480
SCALE_X = DISPLAY_W / SEND_W
SCALE_Y = DISPLAY_H / SEND_H

# Ancho máximo de ventana. Ajusta si tu monitor es más pequeño.
MAX_DISPLAY_W = 900

# --- Intervalos de envío ---
INTERVALO_RAPIDO_SEG  = 0.10   # ~10 FPS → solo YOLO face (sin face_encoding). Objetivo: lag < 150ms.
INTERVALO_COMPLETO_SEG = 2.0   # Cada 2 s → pipeline completo (identity + uniforme).

# --- Estado de detección (dos niveles) ---
posiciones_rapidas = []    # Posiciones de caras, actualizado ~10 FPS
cache_identidades  = []    # Identidad + uniforme, actualizado cada 2 s

_lock_posiciones  = threading.Lock()
_lock_identidades = threading.Lock()

# --- Frame sincronizado ---
# Almacena el frame exacto que fue enviado al servidor y cuyo resultado ya llegó.
# La ventana muestra ESTE frame (no el live), garantizando que los cuadros
# siempre coincidan exactamente con lo que se ve en pantalla.
_ultimo_frame_enviado = None   # Frame crudo (full-res) más recientemente enviado al servidor
frame_confirmado      = None   # Frame que corresponde a la última respuesta de detect_boxes_result

# --- UI ---
estado_conexion = "Esperando servidor..."
color_conexion  = (0, 165, 255)
_ultimo_latencia_ms = 0        # Latencia del último ciclo rápido (round-trip)
_t_ultimo_envio_rapido = 0.0   # Timestamp del último envío rápido

sio = socketio.Client(
    reconnection=True,
    reconnection_attempts=0,
    reconnection_delay=1,
    reconnection_delay_max=5,
)


@sio.event
def connect():
    global estado_conexion, color_conexion
    estado_conexion = "Servidor Online"
    color_conexion  = (0, 255, 0)
    print("Conectado a Socket.IO.")


@sio.event
def disconnect():
    global estado_conexion, color_conexion, posiciones_rapidas, cache_identidades, frame_confirmado
    estado_conexion = "Sin conexion al servidor..."
    color_conexion  = (0, 0, 255)
    with _lock_posiciones:
        posiciones_rapidas = []
    with _lock_identidades:
        cache_identidades = []
    frame_confirmado = None
    print("Desconectado de Socket.IO.")


@sio.on("detect_boxes_result")
def handle_boxes_result(datos):
    """
    Respuesta del servidor al detect_boxes (solo YOLO face, ~50-80 ms round-trip).
    1. Actualiza las posiciones de caras.
    2. Fija el 'frame_confirmado' = el frame que fue enviado a ese análisis.
       Así la ventana muestra EXACTAMENTE el frame que YOLO vio → sincronía perfecta.
    3. Si no hay caras, vuelve a video en vivo (frame_confirmado = None).
    """
    global posiciones_rapidas, frame_confirmado, _ultimo_latencia_ms
    nuevas = datos if isinstance(datos, list) else []

    _ultimo_latencia_ms = int((time.time() - _t_ultimo_envio_rapido) * 1000)

    with _lock_posiciones:
        posiciones_rapidas = nuevas

    if nuevas:
        frame_confirmado = _ultimo_frame_enviado   # sincronía cuadro ↔ frame
    else:
        frame_confirmado = None                    # sin cara → video en vivo


@sio.on("detect_results")
def handle_detect_results(datos):
    """Respuesta del pipeline completo (identity + uniforme). Actualiza cache cada ~2 s."""
    global cache_identidades
    nuevas = datos if isinstance(datos, list) else datos.get("students", [])
    with _lock_identidades:
        cache_identidades = nuevas


def conectar_sio_background():
    while True:
        if not sio.connected:
            try:
                sio.connect(URL_SERVIDOR)
                sio.wait()
            except socketio.exceptions.ConnectionError:
                pass
        time.sleep(2)


frame_actual = None


def leer_camara_continuamente():
    global frame_actual
    ruta_video = "./clients/assets/6.MOV"
    cap = cv2.VideoCapture(ruta_video)

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    intervalo = 1.0 / fps_video

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame_actual = frame
        restante = intervalo - (time.time() - t0)
        if restante > 0:
            time.sleep(restante)


def hilo_envio_rapido():
    """
    Envía el frame más reciente al servidor para DETECCIÓN RÁPIDA (solo YOLO face).
    Guarda el frame crudo exacto en `_ultimo_frame_enviado` ANTES del emit,
    de modo que cuando llegue la respuesta podamos mostrarlo sincronizado.
    """
    global _ultimo_frame_enviado, _t_ultimo_envio_rapido
    while True:
        if sio.connected and frame_actual is not None:
            snap = frame_actual.copy()           # copia estable para sincronía
            _ultimo_frame_enviado = snap         # guardar ANTES de emitir
            _t_ultimo_envio_rapido = time.time()
            try:
                resized = cv2.resize(snap, (SEND_W, SEND_H))
                _, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                sio.emit("detect_boxes", {"image": buf.tobytes()})
            except Exception as e:
                print(f"Error emit rapido: {e}")
        time.sleep(INTERVALO_RAPIDO_SEG)


def hilo_envio_completo():
    """Envía frames para el pipeline completo (identity + uniforme) cada 2 s."""
    while True:
        if sio.connected and frame_actual is not None:
            try:
                snap   = frame_actual
                resized = cv2.resize(snap, (SEND_W, SEND_H))
                _, buf  = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                sio.emit("detect_frame", {"image": buf.tobytes()})
            except Exception as e:
                print(f"Error emit completo: {e}")
        time.sleep(INTERVALO_COMPLETO_SEG)


def _identity_mas_cercana(cx, cy, cache):
    """Busca la detección completa más cercana al centro (cx, cy) de una cara."""
    mejor, mejor_dist = None, float("inf")
    for det in cache:
        t, r, b, l = det.get("location", [0, 0, 0, 0])
        det_cx = (int(l * SCALE_X) + int(r * SCALE_X)) // 2
        det_cy = (int(t * SCALE_Y) + int(b * SCALE_Y)) // 2
        dist = ((cx - det_cx) ** 2 + (cy - det_cy) ** 2) ** 0.5
        if dist < mejor_dist:
            mejor_dist, mejor = dist, det
    return mejor


def open_cam():
    global frame_actual, DISPLAY_W, DISPLAY_H, SCALE_X, SCALE_Y

    ruta_video = "./clients/assets/6.MOV"

    # --- Detectar resolución nativa del video ---
    probe = cv2.VideoCapture(ruta_video)
    native_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()

    if native_w > 0 and native_h > 0:
        if native_w > MAX_DISPLAY_W:
            ratio    = MAX_DISPLAY_W / native_w
            DISPLAY_W = MAX_DISPLAY_W
            DISPLAY_H = int(native_h * ratio)
        else:
            DISPLAY_W = native_w
            DISPLAY_H = native_h

    SCALE_X = DISPLAY_W / SEND_W
    SCALE_Y = DISPLAY_H / SEND_H
    print(f"Video: {native_w}x{native_h} → Ventana: {DISPLAY_W}x{DISPLAY_H} | Escala {SCALE_X:.2f}x{SCALE_Y:.2f}")

    # Ventana redimensionable: el usuario puede arrastrar los bordes
    cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(NOMBRE_VENTANA, DISPLAY_W, DISPLAY_H)

    threading.Thread(target=leer_camara_continuamente, daemon=True).start()
    while frame_actual is None:
        time.sleep(0.1)

    threading.Thread(target=hilo_envio_rapido,   daemon=True).start()
    threading.Thread(target=hilo_envio_completo, daemon=True).start()

    while True:
        # ── Elegir el frame a mostrar ──────────────────────────────────────────
        # Si hay frame confirmado (sincronizado con la última detección YOLO),
        # lo mostramos. Los cuadros están garantizados en la posición exacta.
        # Si no hay confirmación (inicio o sin caras), mostramos el video en vivo.
        fc = frame_confirmado
        if fc is not None:
            frame = cv2.resize(fc, (DISPLAY_W, DISPLAY_H))
        else:
            fa = frame_actual
            if fa is None:
                time.sleep(0.01)
                continue
            frame = cv2.resize(fa, (DISPLAY_W, DISPLAY_H))

        # ── HUD ───────────────────────────────────────────────────────────────
        cv2.putText(frame, f"Lat: {_ultimo_latencia_ms}ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, estado_conexion, (DISPLAY_W - 240, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_conexion, 2)

        # ── Leer estado de detección ───────────────────────────────────────────
        with _lock_posiciones:
            pos_render = list(posiciones_rapidas)
        with _lock_identidades:
            cache_render = list(cache_identidades)

        necesita_cuerpo = False

        for pos in pos_render:
            top, right, bottom, left = pos.get("location", [0, 0, 0, 0])
            top    = int(top    * SCALE_Y)
            right  = int(right  * SCALE_X)
            bottom = int(bottom * SCALE_Y)
            left   = int(left   * SCALE_X)

            cx = (left + right)  // 2
            cy = (top  + bottom) // 2

            det            = _identity_mas_cercana(cx, cy, cache_render)
            nombre         = "Detectando..."
            tiene_uniforme = None
            clothing_disp  = []
            needs_body     = False

            if det is not None:
                nombre         = det.get("identity", "Desconocido")
                tiene_uniforme = det.get("has_uniform")
                needs_body     = det.get("needs_full_body_view", False)

                # Delta de posición cache → actual para reubicar clothing boxes
                ct, cr, cb_c, cl = det.get("location", [0, 0, 0, 0])
                cx_cache = (int(cl * SCALE_X) + int(cr * SCALE_X)) // 2
                cy_cache = (int(ct * SCALE_Y) + int(cb_c * SCALE_Y)) // 2
                dx, dy   = cx - cx_cache, cy - cy_cache

                for cb in det.get("clothing_boxes", []):
                    bx1, by1, bx2, by2 = cb.get("box", [0, 0, 0, 0])
                    clothing_disp.append({
                        "class": cb.get("class", "Ropa"),
                        "valid": cb.get("valid", False),
                        "box":  [int(bx1 * SCALE_X) + dx, int(by1 * SCALE_Y) + dy,
                                 int(bx2 * SCALE_X) + dx, int(by2 * SCALE_Y) + dy],
                    })

            # Color del recuadro de cara
            if nombre == "Detectando...":
                color_rec = (180, 180, 180)   # gris
            elif nombre == "Desconocido":
                color_rec = (0, 0, 255)        # rojo
            elif tiene_uniforme:
                color_rec = (0, 255, 0)        # verde
            else:
                color_rec = (0, 165, 255)      # naranja

            cv2.rectangle(frame, (left, top), (right, bottom), color_rec, 2)

            sufijo = ("" if tiene_uniforme is None
                      else (" | UNIFORME: SI" if tiene_uniforme else " | UNIFORME: NO"))
            cv2.putText(frame, f"{nombre}{sufijo}", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_rec, 2)

            for cb in clothing_disp:
                bx1, by1, bx2, by2 = cb["box"]
                c_color = (0, 200, 0) if cb["valid"] else (0, 0, 200)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), c_color, 2)
                cv2.putText(frame, cb["class"].upper(), (bx1, by1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_color, 2)

            if needs_body:
                necesita_cuerpo = True

        if necesita_cuerpo:
            cv2.putText(frame, "ACERQUESE/ALEJESE PARA VER PANTALON",
                        (50, DISPLAY_H - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(NOMBRE_VENTANA, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    threading.Thread(target=conectar_sio_background, daemon=True).start()
    open_cam()
