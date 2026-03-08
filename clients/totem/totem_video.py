import cv2
import socketio
import time
import threading
import numpy as np

URL_SERVIDOR   = "http://localhost:8000"
NOMBRE_VENTANA = "AI Sentinel - En vivo"

# ---------------------------------------------------------------------------
# Detectar tamaño de pantalla
# ---------------------------------------------------------------------------
try:
    import tkinter as _tk
    _root = _tk.Tk(); _root.withdraw()
    _screen_w = _root.winfo_screenwidth()
    _screen_h = _root.winfo_screenheight()
    _root.destroy(); del _tk, _root
    MAX_DISPLAY_W = int(_screen_w * 0.88)
    MAX_DISPLAY_H = int(_screen_h * 0.88)
except Exception:
    MAX_DISPLAY_W = 480
    MAX_DISPLAY_H = 854

# ---------------------------------------------------------------------------
# Diagnóstico CUDA (informativo, sin error si torch no instalado)
# ---------------------------------------------------------------------------
try:
    import torch as _torch
    if _torch.cuda.is_available():
        print(f"[Cliente] GPU: {_torch.cuda.get_device_name(0)} (inferencia en SERVIDOR)")
    del _torch
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Resoluciones (se recalculan en open_cam según el video real)
# ---------------------------------------------------------------------------
SEND_W, SEND_H = 180, 320   # portrait por defecto
DISPLAY_W      = 480
DISPLAY_H      = 854
SCALE_X        = DISPLAY_W / SEND_W
SCALE_Y        = DISPLAY_H / SEND_H

# ---------------------------------------------------------------------------
# Intervalos
# ---------------------------------------------------------------------------
INTERVALO_COMPLETO_SEG = 1.5   # Pipeline identidad + uniforme
FPS_RENDER_MAX         = 30    # Cap del loop de render

# ---------------------------------------------------------------------------
# Kalman Filter tracker
# ---------------------------------------------------------------------------
# Cada cara detectada se convierte en un FaceTrack con su propio Kalman filter
# 6-dimensional: estado = [cx, cy, w, h, vx, vy]
#
# El tracker corre a 30 FPS (render loop). El servidor responde a ~10 FPS.
# Entre respuestas, el Kalman extrapola la posición usando la velocidad estimada.
# → Las cajas se mueven suavemente con la persona, sin esperar al servidor.
# ---------------------------------------------------------------------------

class FaceTrack:
    """
    Track individual de una cara con Kalman filter 6D.

    Estado:      [cx, cy, w, h, vx, vy]
    Medición:    [cx, cy, w, h]  (del servidor, en coords display)
    Predicción:  cx += vx*dt  (extrapolación lineal de velocidad)

    Parámetros clave:
      Q (process noise):      qué tanto cambia el estado entre frames.
                              Velocidad con ruido bajo → predicción suave.
      R (measurement noise):  qué tanto confiamos en el servidor.
                              Bajo → seguimos al servidor de cerca.
    """
    _lock   = threading.Lock()
    _seq    = 0

    def __init__(self, cx: float, cy: float, w: float, h: float, server_id=None):
        with FaceTrack._lock:
            FaceTrack._seq += 1
            self.local_id  = FaceTrack._seq
        self.server_id     = server_id  # ID de ByteTrack del servidor
        self.identity_data = None       # Resultado del pipeline completo
        self.missed        = 0          # Frames consecutivos sin medición
        self.age           = 0

        # Kalman filter
        dt = 1.0 / FPS_RENDER_MAX
        self.kf = cv2.KalmanFilter(6, 4)

        # F: transición (pos += vel * dt)
        F = np.eye(6, dtype=np.float32)
        F[0, 4] = dt
        F[1, 5] = dt
        self.kf.transitionMatrix = F

        # H: observamos cx, cy, w, h
        self.kf.measurementMatrix = np.eye(4, 6, dtype=np.float32)

        # Q: ruido de proceso — velocidad con poca incertidumbre → predicción estable
        self.kf.processNoiseCov = np.diag(
            [2.0, 2.0, 1.5, 1.5, 0.05, 0.05]
        ).astype(np.float32)

        # R: ruido de medición — confiamos bastante en YOLO (GPU, alta precisión)
        self.kf.measurementNoiseCov = np.diag(
            [3.0, 3.0, 8.0, 8.0]
        ).astype(np.float32)

        # Covarianza inicial alta → acepta corrección rápida en los primeros frames
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 50.0
        self.kf.statePost    = np.array(
            [[cx], [cy], [w], [h], [0.0], [0.0]], dtype=np.float32
        )

    def predict(self) -> tuple[int, int, int, int]:
        """
        Avanza el Kalman un step y devuelve (x1, y1, x2, y2) en display coords.
        Llamar UNA VEZ por frame de render.
        """
        p  = self.kf.predict()  # OpenCV: estado (6,1) o (6,) según versión
        flat = np.asarray(p).flat
        cx = float(flat[0]); cy = float(flat[1])
        w  = max(float(flat[2]), 20.0)
        h  = max(float(flat[3]), 20.0)
        self.missed += 1
        self.age    += 1
        return (int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2))

    def correct(self, cx: float, cy: float, w: float, h: float):
        """Incorpora una nueva medición del servidor (corrige la predicción)."""
        self.kf.correct(np.array([[cx], [cy], [w], [h]], dtype=np.float32))
        self.missed = 0

    def center(self) -> tuple[float, float]:
        s = np.asarray(self.kf.statePost).flat
        return float(s[0]), float(s[1])

    def velocity(self) -> tuple[float, float]:
        s = np.asarray(self.kf.statePost).flat
        return float(s[4]), float(s[5])


class FaceTracker:
    """
    Gestor de múltiples FaceTracks.

    Flujo por cada respuesta del servidor (fast channel):
      1. Matching por server_id (ByteTrack ID) — O(n), más confiable.
      2. Matching residual por distancia euclidiana — para caras sin ID.
      3. Crear nuevos tracks para detecciones sin match.
      4. Incrementar 'missed' en tracks sin actualización.
      5. Podar tracks con missed > MAX_MISSED.

    Flujo en el render loop (30 FPS):
      - predict_all() → avanza Kalman de cada track → posiciones suaves.

    Flujo al llegar resultados del pipeline completo (slow channel):
      - associate_identity() → asigna identity_data al track más cercano.
    """
    MAX_MISSED = 12   # frames sin medición → eliminar track (~400 ms a 30 FPS)
    MAX_DIST   = 90   # px en display para asociar detección → track existente

    def __init__(self):
        self.tracks: list[FaceTrack] = []

    def update(self, detections: list[dict]):
        """
        detections: lista de dicts con keys:
          "location_display": (x1, y1, x2, y2) en coords display
          "server_id":        int | None (ByteTrack ID del servidor)
        """
        if not detections:
            for t in self.tracks:
                t.missed += 1
            self._prune()
            return

        # Preparar datos de detección
        det_data = []
        for d in detections:
            x1, y1, x2, y2 = d["location_display"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w  = float(x2 - x1)
            h  = float(y2 - y1)
            det_data.append((cx, cy, w, h, d.get("server_id")))

        matched_t = set()   # id(track) ya emparejados
        matched_d = set()   # índices de detección ya emparejados

        # --- Paso 1: match por server_id (más confiable, O(N)) ---
        sid_map = {t.server_id: t for t in self.tracks if t.server_id is not None}
        for di, (cx, cy, w, h, sid) in enumerate(det_data):
            if sid is not None and sid in sid_map:
                track = sid_map[sid]
                if id(track) not in matched_t:
                    track.correct(cx, cy, w, h)
                    track.server_id = sid  # refrescar por si cambió
                    matched_t.add(id(track))
                    matched_d.add(di)

        # --- Paso 2: match residual por distancia euclidiana ---
        rem_dets   = [i for i in range(len(det_data)) if i not in matched_d]
        rem_tracks = [t for t in self.tracks if id(t) not in matched_t]

        for track in rem_tracks:
            if not rem_dets:
                break
            t_cx, t_cy = track.center()
            best_i, best_d = None, self.MAX_DIST
            for i in rem_dets:
                cx, cy, *_ = det_data[i]
                dist = ((cx - t_cx) ** 2 + (cy - t_cy) ** 2) ** 0.5
                if dist < best_d:
                    best_d, best_i = dist, i
            if best_i is not None:
                cx, cy, w, h, sid = det_data[best_i]
                track.correct(cx, cy, w, h)
                if sid is not None:
                    track.server_id = sid
                matched_t.add(id(track))
                rem_dets.remove(best_i)

        # --- Paso 3: nuevos tracks para detecciones sin match ---
        for i in rem_dets:
            cx, cy, w, h, sid = det_data[i]
            self.tracks.append(FaceTrack(cx, cy, w, h, server_id=sid))

        # --- Paso 4: incrementar missed en tracks sin actualización ---
        for t in self.tracks:
            if id(t) not in matched_t:
                t.missed += 1

        self._prune()

    def predict_all(self) -> list[tuple[tuple, FaceTrack]]:
        """
        Avanza el Kalman de cada track activo y devuelve sus posiciones predichas.
        Llamar UNA VEZ por frame de render.
        """
        return [
            (t.predict(), t)
            for t in self.tracks
            if t.missed <= self.MAX_MISSED
        ]

    def associate_identity(self, results: list[dict]):
        """
        Asigna los resultados del pipeline completo (identidad + uniforme) al
        track más cercano. Si el track ya tiene identidad para la misma persona,
        solo la actualiza (no resetea clothing si ya era conocida).
        """
        for res in results:
            loc_d = res.get("location_display")
            if loc_d is None:
                continue
            x1, y1, x2, y2 = loc_d
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            best_t, best_d = None, 120.0
            for t in self.tracks:
                t_cx, t_cy = t.center()
                d = ((cx - t_cx) ** 2 + (cy - t_cy) ** 2) ** 0.5
                if d < best_d:
                    best_d, best_t = d, t

            if best_t is not None:
                best_t.identity_data = res

    def clear(self):
        self.tracks.clear()

    def _prune(self):
        self.tracks = [t for t in self.tracks if t.missed <= self.MAX_MISSED]


# ---------------------------------------------------------------------------
# Estado global
# ---------------------------------------------------------------------------
face_tracker   = FaceTracker()
_lock_tracker  = threading.Lock()

# Control de flujo del canal rápido (ack-based)
_envio_rapido_pendiente = False
_t_ultimo_envio_rapido  = 0.0
_ultimo_latencia_ms     = 0

# UI
estado_conexion = "Esperando servidor..."
color_conexion  = (0, 165, 255)

# FPS del render loop
_fps_contador = 0
_fps_ts       = time.time()
_fps_actual   = 0.0

frame_actual = None

# ---------------------------------------------------------------------------
# Socket.IO client
# ---------------------------------------------------------------------------
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
    global estado_conexion, color_conexion, _envio_rapido_pendiente
    estado_conexion         = "Sin conexion al servidor..."
    color_conexion          = (0, 0, 255)
    _envio_rapido_pendiente = False
    with _lock_tracker:
        face_tracker.clear()
    print("Desconectado de Socket.IO.")


def _loc_to_display(loc):
    """Convierte (top,right,bottom,left) en coords SEND → (x1,y1,x2,y2) en display."""
    top, right, bottom, left = loc
    return (
        int(left   * SCALE_X),
        int(top    * SCALE_Y),
        int(right  * SCALE_X),
        int(bottom * SCALE_Y),
    )


@sio.on("detect_boxes_result")
def handle_boxes_result(datos):
    """
    Respuesta del canal rápido (YOLO-Face + ByteTrack).
    1. Libera el flag ack → el hilo emisor puede enviar el siguiente frame.
    2. Convierte locaciones a coordenadas display.
    3. Llama face_tracker.update() → el Kalman de cada track se corrige
       con la medición real, mientras sigue prediciendo entre frames.
    """
    global _ultimo_latencia_ms, _envio_rapido_pendiente
    _envio_rapido_pendiente = False
    _ultimo_latencia_ms     = int((time.time() - _t_ultimo_envio_rapido) * 1000)

    nuevas = datos if isinstance(datos, list) else []
    dets   = [
        {
            "location_display": _loc_to_display(d.get("location", [0, 0, 0, 0])),
            "server_id":        d.get("track_id"),
        }
        for d in nuevas
    ]
    with _lock_tracker:
        face_tracker.update(dets)


@sio.on("detect_results")
def handle_detect_results(datos):
    """
    Respuesta del pipeline completo (identidad + uniforme).
    Pre-escala todas las coordenadas a display y las asocia a los tracks activos.
    """
    nuevas = datos if isinstance(datos, list) else datos.get("students", [])
    enriched = []
    for res in nuevas:
        loc   = res.get("location", [0, 0, 0, 0])
        loc_d = _loc_to_display(loc)

        # Escalar clothing boxes a display coords
        c_boxes_d = []
        for cb in res.get("clothing_boxes", []):
            bx1, by1, bx2, by2 = cb["box"]
            c_boxes_d.append({
                "class": cb["class"],
                "valid": cb.get("valid", False),
                "box_display": [
                    int(bx1 * SCALE_X), int(by1 * SCALE_Y),
                    int(bx2 * SCALE_X), int(by2 * SCALE_Y),
                ],
            })

        enriched.append({
            **res,
            "location_display":    loc_d,
            "clothing_boxes_disp": c_boxes_d,
        })

    with _lock_tracker:
        face_tracker.associate_identity(enriched)


# ---------------------------------------------------------------------------
# Hilo de reconexión Socket.IO
# ---------------------------------------------------------------------------
def conectar_sio_background():
    while True:
        if not sio.connected:
            try:
                sio.connect(URL_SERVIDOR)
                sio.wait()
            except socketio.exceptions.ConnectionError:
                pass
            except Exception as e:
                print(f"[Socket] Error: {e}")
        time.sleep(2)


# ---------------------------------------------------------------------------
# Hilo lector de cámara
# ---------------------------------------------------------------------------
def leer_camara_continuamente():
    """Lee frames del video y los almacena pre-resizeados a DISPLAY_W x DISPLAY_H."""
    global frame_actual
    ruta_video  = "./assets/3.MOV"
    cap         = cv2.VideoCapture(ruta_video)
    fps_video   = cap.get(cv2.CAP_PROP_FPS) or 30
    intervalo   = 1.0 / fps_video

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame_actual = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        restante = intervalo - (time.time() - t0)
        if restante > 0:
            time.sleep(restante)


# ---------------------------------------------------------------------------
# Hilo emisor: canal rápido (ack-based)
# ---------------------------------------------------------------------------
def hilo_envio_rapido():
    """
    Envía frames al servidor para detección rápida con control de flujo ack-based.
    Solo envía cuando la respuesta anterior llegó (o timeout 1 s).
    La tasa real se adapta automáticamente a la latencia del servidor.
    """
    global _t_ultimo_envio_rapido, _envio_rapido_pendiente
    while True:
        ahora   = time.time()
        timeout = (ahora - _t_ultimo_envio_rapido) > 1.0

        if sio.connected and frame_actual is not None and (not _envio_rapido_pendiente or timeout):
            _envio_rapido_pendiente = True
            _t_ultimo_envio_rapido  = ahora
            try:
                resized = cv2.resize(frame_actual, (SEND_W, SEND_H))
                _, buf  = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                sio.emit("detect_boxes", {"image": buf.tobytes()})
            except Exception as e:
                _envio_rapido_pendiente = False
                print(f"Error emit rapido: {e}")
        time.sleep(0.02)


# ---------------------------------------------------------------------------
# Hilo emisor: pipeline completo (identidad + uniforme)
# ---------------------------------------------------------------------------
def hilo_envio_completo():
    """
    Envía frames para el pipeline completo cada INTERVALO_COMPLETO_SEG.
    Solo envía si hay tracks activos (hay personas en cámara).
    """
    while True:
        with _lock_tracker:
            hay_tracks = len(face_tracker.tracks) > 0

        if sio.connected and frame_actual is not None and hay_tracks:
            try:
                resized = cv2.resize(frame_actual, (SEND_W, SEND_H))
                _, buf  = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                sio.emit("detect_frame", {"image": buf.tobytes()})
            except Exception as e:
                print(f"Error emit completo: {e}")
        time.sleep(INTERVALO_COMPLETO_SEG)


# ---------------------------------------------------------------------------
# Loop principal de render
# ---------------------------------------------------------------------------
def open_cam():
    global frame_actual, DISPLAY_W, DISPLAY_H, SCALE_X, SCALE_Y, SEND_W, SEND_H
    global _fps_contador, _fps_ts, _fps_actual

    ruta_video = "./assets/3.MOV"

    # Resolución nativa del video
    probe    = cv2.VideoCapture(ruta_video)
    native_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()

    scale = 1.0
    if native_w > 0 and native_h > 0:
        scale     = min(MAX_DISPLAY_W / native_w, MAX_DISPLAY_H / native_h, 1.0)
        DISPLAY_W = int(native_w * scale)
        DISPLAY_H = int(native_h * scale)

    # SEND dinámico: mismo aspect ratio que el video (crítico para SCALE_X ≈ SCALE_Y)
    SEND_LONG = 320
    if DISPLAY_H >= DISPLAY_W:
        SEND_H = SEND_LONG
        SEND_W = max(1, int(round(DISPLAY_W * SEND_LONG / DISPLAY_H)))
    else:
        SEND_W = SEND_LONG
        SEND_H = max(1, int(round(DISPLAY_H * SEND_LONG / DISPLAY_W)))

    SCALE_X = DISPLAY_W / SEND_W
    SCALE_Y = DISPLAY_H / SEND_H
    ok_str  = "✅" if abs(SCALE_X - SCALE_Y) < 0.1 else "⚠️"
    print(f"[Ventana] {native_w}x{native_h} → {DISPLAY_W}x{DISPLAY_H} "
          f"| SEND {SEND_W}x{SEND_H} | SCALE {SCALE_X:.2f}x{SCALE_Y:.2f} {ok_str}")

    cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(NOMBRE_VENTANA, DISPLAY_W, DISPLAY_H)

    threading.Thread(target=leer_camara_continuamente, daemon=True).start()
    while frame_actual is None:
        time.sleep(0.05)

    threading.Thread(target=hilo_envio_rapido,   daemon=True).start()
    threading.Thread(target=hilo_envio_completo, daemon=True).start()

    _ms_por_frame = max(1, int(1000 / FPS_RENDER_MAX))

    while True:
        # ── Frame en vivo (siempre) ────────────────────────────────────────
        fa = frame_actual
        if fa is None:
            time.sleep(0.01)
            continue
        frame = fa.copy()

        # ── Contador FPS ──────────────────────────────────────────────────
        _fps_contador += 1
        _dt = time.time() - _fps_ts
        if _dt >= 1.0:
            _fps_actual   = _fps_contador / _dt
            _fps_contador = 0
            _fps_ts       = time.time()

        # ── HUD ───────────────────────────────────────────────────────────
        cv2.putText(frame, f"FPS: {_fps_actual:.1f} | Lat: {_ultimo_latencia_ms}ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(frame, estado_conexion,
                    (DISPLAY_W - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_conexion, 2)

        # ── Obtener posiciones predichas por Kalman (thread-safe snapshot) ─
        with _lock_tracker:
            tracked = face_tracker.predict_all()

        necesita_cuerpo = False

        for box_pred, track in tracked:
            # ── Clamping a los bordes del frame ──────────────────────────
            x1 = max(0, min(box_pred[0], DISPLAY_W - 1))
            y1 = max(0, min(box_pred[1], DISPLAY_H - 1))
            x2 = max(0, min(box_pred[2], DISPLAY_W - 1))
            y2 = max(0, min(box_pred[3], DISPLAY_H - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            det            = track.identity_data
            nombre         = "Detectando..."
            tiene_uniforme = None
            clothing_items = []
            needs_body     = False

            if det:
                nombre         = det.get("identity", "Desconocido")
                tiene_uniforme = det.get("has_uniform")
                needs_body     = det.get("needs_full_body_view", False)

                # Delta de posición: mueve clothing boxes con el track actual.
                # Cuando el servidor detectó la ropa, la cara estaba en loc_d;
                # ahora el Kalman predice que está en (cx_now, cy_now).
                # Desplazar las boxes por esa diferencia → ropa sigue al cuerpo.
                loc_d = det.get("location_display", (x1, y1, x2, y2))
                cx_old = (loc_d[0] + loc_d[2]) / 2.0
                cy_old = (loc_d[1] + loc_d[3]) / 2.0
                cx_now = (x1 + x2) / 2.0
                cy_now = (y1 + y2) / 2.0
                dx = int(cx_now - cx_old)
                dy = int(cy_now - cy_old)

                for cb in det.get("clothing_boxes_disp", []):
                    bx1, by1, bx2, by2 = cb["box_display"]
                    clothing_items.append({
                        "class": cb["class"],
                        "valid": cb.get("valid", False),
                        "box":  (bx1 + dx, by1 + dy, bx2 + dx, by2 + dy),
                    })

            # ── Color del recuadro ──────────────────────────────────────
            if nombre == "Detectando...":
                color_rec = (180, 180, 180)   # gris
            elif nombre == "Desconocido":
                color_rec = (0, 0, 255)        # rojo
            elif tiene_uniforme:
                color_rec = (0, 255, 0)        # verde
            else:
                color_rec = (0, 165, 255)      # naranja

            # Recuadro de cara
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_rec, 2)

            sufijo = ("" if tiene_uniforme is None
                      else (" | UNIFORME: SI" if tiene_uniforme else " | UNIFORME: NO"))
            cv2.putText(frame, f"{nombre}{sufijo}",
                        (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_rec, 2)

            # Cajas de ropa (con delta aplicado)
            for cb in clothing_items:
                bx1, by1, bx2, by2 = cb["box"]
                c_color = (0, 200, 0) if cb["valid"] else (0, 0, 200)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), c_color, 2)
                cv2.putText(frame, cb["class"].upper(),
                            (bx1, max(by1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_color, 2)

            if needs_body:
                necesita_cuerpo = True

        if necesita_cuerpo:
            cv2.putText(frame, "ACERQUESE/ALEJESE PARA VER PANTALON",
                        (50, DISPLAY_H - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(NOMBRE_VENTANA, frame)
        if cv2.waitKey(_ms_por_frame) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    threading.Thread(target=conectar_sio_background, daemon=True).start()
    open_cam()
