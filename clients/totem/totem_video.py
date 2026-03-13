import cv2
import socketio
import time
import threading
import numpy as np
import os
import sys
import requests
import base64
import time

# Agregar ruta para que ui_utils.py pueda ser importado
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ui_utils import draw_futuristic_box

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
SEND_W, SEND_H = 180, 320   # canal rápido (detect_boxes): portrait por defecto
DISPLAY_W      = 480
DISPLAY_H      = 854
SCALE_X        = DISPLAY_W / SEND_W
SCALE_Y        = DISPLAY_H / SEND_H

# Canal lento (detect_frame / identidad + uniforme):
#   2× resolución del canal rápido → caras ~54px en lugar de ~27px.
#   dlib (face_recognition) recomienda ≥40px de cara para encoding confiable.
#   Con mayor resolución, face_recognition extrae mejores embeddings y la
#   detección de ropa por YOLO también mejora (usa imgsz=640 nativo).
#   Escala inversa SCALE_SLOW = DISPLAY / SEND_SLOW (más pequeña que SCALE).
SEND_W_SLOW  = 180 * 2  # se recalcula en open_cam igual que SEND_W
SEND_H_SLOW  = 320 * 2
SCALE_X_SLOW = DISPLAY_W / SEND_W_SLOW
SCALE_Y_SLOW = DISPLAY_H / SEND_H_SLOW

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
        self.server_id     = server_id
        self.identity_data = None
        self.missed        = 0
        self.age           = 0
        # Timestamp hasta el cual el uniforme está "congelado" como completo.
        # Si time.time() < uniform_confirmed_until → no re-enviar frame al servidor.
        self.uniform_confirmed_until = 0.0
        self._last_ts      = time.monotonic()   # timestamp del último predict()

        # Kalman filter 6D: [cx, cy, w, h, vx, vy]
        # dt es VARIABLE: se actualiza en cada predict() con el tiempo real transcurrido.
        # dt fijo = 1/30s causaba acumulación de error cuando el render no era exactamente
        # a 30 FPS (GIL de Python, contención de locks) → lag visible en personas rápidas.
        self.kf = cv2.KalmanFilter(6, 4)

        # F se inicializa con dt≈0; se sobreescribe en cada predict() con dt real
        self.kf.transitionMatrix  = np.eye(6, dtype=np.float32)
        self.kf.measurementMatrix = np.eye(4, 6, dtype=np.float32)

        # Q: ruido de proceso.
        # Q_vel=0.10 (reducido desde 0.35): menos "vuelo" cuando la persona
        # desacelera o para. El Kalman no intenta seguir velocidades ficticias.
        # Q_pos=1.5: suficiente para absorber pequeñas discrepancias de medición.
        self.kf.processNoiseCov = np.diag(
            [1.5, 1.5, 0.8, 0.8, 0.10, 0.10]
        ).astype(np.float32)

        # R: ruido de medición.
        # R=2.0 (reducido desde 4.0): confiamos más en YOLO+ByteTrack GPU.
        # El filtro "salta" más rápido a la medición real → menos desviación.
        self.kf.measurementNoiseCov = np.diag(
            [2.0, 2.0, 6.0, 6.0]
        ).astype(np.float32)

        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 50.0
        self.kf.statePost    = np.array(
            [[cx], [cy], [w], [h], [0.0], [0.0]], dtype=np.float32
        )

    def predict(self) -> tuple[int, int, int, int]:
        """
        Avanza el Kalman con dt REAL desde la última llamada.

        Por qué dt real es importante:
        - El GIL de Python y la contención en _lock_tracker hacen que el render
          loop no sea exactamente 30 FPS. Con dt fijo la predicción de posición
          diverge cuando los frames tardan más/menos de 1/30 s.
        - Con dt real: si tardó 45 ms desde el último predict, la caja se mueve
          exactamente vx*0.045 píxeles — sin acumulación de error.
        """
        now = time.monotonic()
        dt  = min(now - self._last_ts, 0.1)  # cap a 100 ms para no saltar si hay pausa
        self._last_ts = now

        # Actualizar F con dt real antes de predecir
        self.kf.transitionMatrix[0, 4] = dt
        self.kf.transitionMatrix[1, 5] = dt

        p    = self.kf.predict()
        flat = np.asarray(p).flat
        cx   = float(flat[0]); cy = float(flat[1])
        w    = max(float(flat[2]), 20.0)
        h    = max(float(flat[3]), 20.0)
        self.missed += 1
        self.age    += 1
        return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))

    def correct(self, cx: float, cy: float, w: float, h: float):
        """
        Incorpora una nueva medición del servidor y amortigua la velocidad.

        Por qué amortiguar velocidad después de correct():
        Cuando el servidor envía una posición real (ground truth), la velocidad
        estimada por el Kalman podría estar desviada (ej: el tracker predijo que
        la persona iba a seguir moviéndose pero se detuvo). Sin amortiguación,
        el box "vuela" más allá del punto de corrección. Con 0.6×, la velocidad
        se reduce parcialmente → el box queda cerca del ground truth del servidor.
        """
        self.kf.correct(np.array([[cx], [cy], [w], [h]], dtype=np.float32))
        # Amortiguar velocidad post-corrección para evitar overshooting
        s = self.kf.statePost
        s[4, 0] *= 0.6   # vx
        s[5, 0] *= 0.6   # vy
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
    MAX_MISSED = 30   # frames sin medición → eliminar track (1 segundo a 30 FPS, antes 12)
    MAX_DIST   = 150  # px en display para asociar detección → track existente (rápido, antes 90)

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
        track más cercano.
        Si el uniforme queda completo, congela la verificación 30 segundos:
        uniform_confirmed_until = now + 30  → hilo_envio_completo omite ese track.
        """
        now = time.time()
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
                if res.get("has_uniform"):
                    # Uniforme completo → no re-verificar por 30 s
                    best_t.uniform_confirmed_until = now + 30.0
                elif res.get("identity") not in ("Desconocido", "Detectando...", None):
                    # Uniforme incompleto para persona conocida → resetear lock
                    best_t.uniform_confirmed_until = 0.0

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


def _loc_to_display(loc, sx=None, sy=None):
    """
    Convierte (top,right,bottom,left) en coords SEND → (x1,y1,x2,y2) en display.
    sx/sy opcionales: escala del canal que generó la detección.
      - Sin parámetros:  canal rápido (SCALE_X, SCALE_Y)
      - sx=SCALE_X_SLOW: canal lento (frame 2× más grande → escala 2× más pequeña)
    """
    top, right, bottom, left = loc
    _sx = sx if sx is not None else SCALE_X
    _sy = sy if sy is not None else SCALE_Y
    return (
        int(left   * _sx),
        int(top    * _sy),
        int(right  * _sx),
        int(bottom * _sy),
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
    Pre-escala con SCALE_SLOW (canal lento = 2× resolución que el canal rápido).
    """
    nuevas = datos if isinstance(datos, list) else datos.get("students", [])
    enriched = []
    for res in nuevas:
        loc   = res.get("location", [0, 0, 0, 0])
        loc_d = _loc_to_display(loc, sx=SCALE_X_SLOW, sy=SCALE_Y_SLOW)

        # Clothing boxes también vienen en coordenadas del canal lento (2×)
        c_boxes_d = []
        for cb in res.get("clothing_boxes", []):
            bx1, by1, bx2, by2 = cb["box"]
            c_boxes_d.append({
                "class": cb["class"],
                "valid": cb.get("valid", False),
                "box_display": [
                    int(bx1 * SCALE_X_SLOW), int(by1 * SCALE_Y_SLOW),
                    int(bx2 * SCALE_X_SLOW), int(by2 * SCALE_Y_SLOW),
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
_ultimo_envio_completo = 0.0

def hilo_envio_completo():
    """
    Envía frames para el pipeline completo (identidad + uniforme) en 2× resolución.

    Canal lento usa SEND_W_SLOW × SEND_H_SLOW (el doble del canal rápido) y
    calidad JPEG 92% para maximizar la información facial disponible para dlib.

    Optimización de uniforme completo (30 s):
      Si todos los tracks activos ya tienen uniforme confirmado (has_uniform=True) y
      el bloqueo temporal aún no expiró, NO se envía el frame → el servidor no procesa
      YOLO+ResNet+ChromaDB para esa persona innecesariamente.
      El recuadro sigue verde en el cliente gracias al identity_data cacheado.
    """
    global _ultimo_envio_completo
    while True:
        with _lock_tracker:
            hay_tracks = len(face_tracker.tracks) > 0
            now = time.time()
            
            # ¿Hay rostros que acaban de entrar y no tienen NINGÚN dato?
            hay_urgentes = any(
                t.identity_data is None
                for t in face_tracker.tracks
                if t.missed <= FaceTracker.MAX_MISSED
            )
            
            # ¿Hay algún track activo que necesite verificación?
            hay_pendientes = any(
                t.identity_data is None
                or t.identity_data.get("identity") in ("Detectando...", None)
                or now > t.uniform_confirmed_until
                for t in face_tracker.tracks
                if t.missed <= FaceTracker.MAX_MISSED
            )

        # Si hay personas nuevas, reducir el intervalo a 0.2s para que se procesen rápido.
        # De lo contrario, usar el intervalo normal (1.5s).
        intervalo_actual = 0.2 if hay_urgentes else INTERVALO_COMPLETO_SEG

        if sio.connected and frame_actual is not None and hay_tracks and hay_pendientes:
            if now - _ultimo_envio_completo >= intervalo_actual:
                _ultimo_envio_completo = now
                try:
                    resized = cv2.resize(frame_actual, (SEND_W_SLOW, SEND_H_SLOW))
                    _, buf  = cv2.imencode(".jpg", resized,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    sio.emit("detect_frame", {"image": buf.tobytes()})
                except Exception as e:
                    print(f"Error emit completo: {e}")
        
        # Sleep muy corto para no bloquear y reaccionar instantáneamente
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# Loop principal de render
# ---------------------------------------------------------------------------
def open_cam():
    global frame_actual, DISPLAY_W, DISPLAY_H, SCALE_X, SCALE_Y, SEND_W, SEND_H
    global SEND_W_SLOW, SEND_H_SLOW, SCALE_X_SLOW, SCALE_Y_SLOW
    global _fps_contador, _fps_ts, _fps_actual
    url_alerts = "http://localhost:3067/AISentinelAdmin/v1/alerts/automatic-detection"
    ultimo_envio_alerta = 0 
    intervalo_alerta = 10

    ruta_video = "./assets/3.MOV"

    if not os.path.exists(ruta_video):
        print(f"\n[ERROR CRÍTICO] El archivo de video '{ruta_video}' no fue encontrado.")
        print("-> Asegurate de que exista o utiliza 'totem_camera.py' para usar la webcam viva.\n")
        sys.exit(1)

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

    # Canal lento: 2× resolución para mejor face encoding (dlib necesita ≥40px/cara)
    SEND_W_SLOW  = SEND_W * 2
    SEND_H_SLOW  = SEND_H * 2
    SCALE_X_SLOW = DISPLAY_W / SEND_W_SLOW
    SCALE_Y_SLOW = DISPLAY_H / SEND_H_SLOW

    ok_str  = "✅" if abs(SCALE_X - SCALE_Y) < 0.1 else "⚠️"
    print(f"[Ventana] {native_w}x{native_h} → {DISPLAY_W}x{DISPLAY_H} "
          f"| SEND_FAST {SEND_W}x{SEND_H} | SEND_SLOW {SEND_W_SLOW}x{SEND_H_SLOW} "
          f"| SCALE {SCALE_X:.2f} {ok_str}")

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
            if det:
                nombre         = det.get("identity", "Desconocido")
                tiene_uniforme = det.get("has_uniform")
                id_card = det.get("student_id")
                tiempo_actual = time.time()
                
                if id_card and nombre != "Desconocido":
                    if (tiempo_actual - ultimo_envio_alerta > intervalo_alerta):
                        try:
                            payload_asistencia = {
                                "idCard": id_card,
                            }
                            
                            def enviar_asistencia(p):
                                try:
                                    requests.post("http://localhost:3067/AISentinelAdmin/v1/attendance/automatic-detection", 
                                                json=p, timeout=2.0)
                                    print(f"📍 Asistencia automática enviada: {p['idCard']}")
                                except Exception as e:
                                    print(f"❌ Error enviando asistencia: {e}")

                            threading.Thread(target=enviar_asistencia, args=(payload_asistencia,), daemon=True).start()
                            
                            ultimo_envio_alerta = tiempo_actual 
                        except Exception as e:
                            print(f"Error preparando frame: {e}")
                
                tiene_accesorio = any(
                    cb.get("class", "").upper() == "ACCESORIO" 
                    for cb in det.get("clothing_boxes_disp", [])
                )
                
                # --- LÓGICA DE ALERTA MEJORADA ---
                
                # Definimos los motivos
                motivo = None
                if tiene_uniforme is False:
                    motivo = "UNIFORME_INCOMPLETO"
                elif tiene_accesorio:
                    motivo = "ACCESORIO_NO_PERMITIDO"

                if motivo and id_card and (tiempo_actual - ultimo_envio_alerta > intervalo_alerta):
                    try:
                        _, buffer_img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                        img_base64 = base64.b64encode(buffer_img).decode('utf-8')

                        payload = {
                            "idCard": id_card, 
                            "has_uniform": tiene_uniforme,
                            "has_accessory": tiene_accesorio,
                            "reason": motivo, 
                            "image": img_base64
                        }
                        
                        def enviar_peticion(p):
                            try:
                                requests.post(url_alerts, json=p, timeout=3.0)
                                print(f"Alerta Enviada ({p['reason']}): {id_card}")
                            except Exception as e:
                                print(f" Error HTTP: {e}")

                        threading.Thread(target=enviar_peticion, args=(payload,), daemon=True).start()
                        
                        ultimo_envio_alerta = tiempo_actual 
                    except Exception as e:
                        print(f"❌ Error preparando alerta: {e}")
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

            # Recuadro de cara futurista
            sufijo = ("" if tiene_uniforme is None
                      else (" | UNIFORME: SI" if tiene_uniforme else " | UNIFORME: NO"))
            etiqueta = f"{nombre}{sufijo}"
            draw_futuristic_box(frame, x1, y1, x2, y2, color_rec, text=etiqueta, text_size=0.6)

            # Cajas de ropa futuristas
            for cb in clothing_items:
                bx1, by1, bx2, by2 = cb["box"]
                if cb["class"].upper() == "ACCESORIO":
                    c_color = (0, 0, 255)   # accesorios siempre en rojo
                elif cb["valid"]:
                    c_color = (0, 200, 0)   # prenda válida → verde
                else:
                    c_color = (0, 100, 255) # prenda inválida → naranja
                
                # Borde más delgado y elegante para cajas de ropa interna
                draw_futuristic_box(frame, bx1, by1, bx2, by2, c_color, text=cb["class"].upper(), text_size=0.5)

        cv2.imshow(NOMBRE_VENTANA, frame)
        if cv2.waitKey(_ms_por_frame) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    threading.Thread(target=conectar_sio_background, daemon=True).start()
    open_cam()