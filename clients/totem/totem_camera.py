import cv2
import socketio
import time
import threading
import base64

URL_SERVIDOR = "http://localhost:8000"

# Variables globales para UI fluida
detecciones_actuales = []
estado_conexion = "Esperando servidor..."
color_conexion = (0, 165, 255)  # Naranja por defecto

# Flag de control para el hilo de envío (evitar acumular frames)
enviando = False
_lock_enviando = threading.Lock()

sio = socketio.Client(reconnection=True, reconnection_attempts=0, reconnection_delay=1, reconnection_delay_max=5)

@sio.event
def connect():
    global estado_conexion, color_conexion
    estado_conexion = "Servidor Online"
    color_conexion = (0, 255, 0)  # Verde
    print("Conectado exitosamente por Socket.IO al servidor FastAPI.")

@sio.event
def disconnect():
    global estado_conexion, color_conexion, detecciones_actuales
    estado_conexion = "Sin conexion al servidor..."
    color_conexion = (0, 0, 255)  # Rojo
    detecciones_actuales = []  # Limpiar cuadros si se cae la red
    print("Desconectado de Socket.IO, esperando reconexión...")

@sio.on('detect_results')
def handle_detect_results(datos):
    """Recibe resultados del servidor y libera el flag de envío."""
    global detecciones_actuales, enviando
    detecciones_actuales = (
        datos if isinstance(datos, list) else datos.get("students", [])
    )
    with _lock_enviando:
        enviando = False

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
    cap = cv2.VideoCapture(1)
    # Reducimos hardware y buffer para eliminar el input lag de OpenCV
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Fundamental: Evitar que opencv almacene fotogramas viejos
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    while True:
        ret, frame = cap.read()
        if ret:
            frame_actual = frame
        time.sleep(0.005) # Libera CPU


def hilo_envio_frames():
    """
    Hilo dedicado exclusivamente a enviar frames al servidor.
    
    Patrón de pipeline desacoplado:
    - El hilo principal dibuja y muestra a 30 FPS sin esperar al servidor.
    - Este hilo envía el frame más reciente SOLO cuando el servidor ya
      contestó el anterior (flag `enviando`). Si el servidor está ocupado
      se descarta el frame viejo y se toma el más reciente al liberar.
    - Esto elimina la cola creciente de frames obsoletos y el wait() bloqueante.
    """
    global enviando
    while True:
        if sio.connected and frame_actual is not None:
            with _lock_enviando:
                if not enviando:
                    # Capturar el frame más fresco en este instante
                    frame_snap = frame_actual
                    enviando = True
                else:
                    time.sleep(0.005)
                    continue

            try:
                frame_envio = cv2.resize(frame_snap, (320, 240))
                _, buffer = cv2.imencode(".jpg", frame_envio, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                sio.emit('detect_frame', {'image': buffer.tobytes()})
            except Exception as e:
                print(f"Error emit: {e}")
                with _lock_enviando:
                    enviando = False  # Liberar si falló el envío
        time.sleep(0.005)


def open_cam():
    global frame_actual
    tiempo_previo_fps = 0
    url_alerts = "http://localhost:3067/AISentinelAdmin/v1/alerts/automatic-detection"
    ultimo_envio_alerta = 0 
    intervalo_alerta = 10

    print("Iniciando sensor de cámara en segundo plano...")
    threading.Thread(target=leer_camara_continuamente, daemon=True).start()

    while frame_actual is None:
        time.sleep(0.1)

    # Hilo dedicado al envío (desacoplado del render)
    threading.Thread(target=hilo_envio_frames, daemon=True).start()

    # Loop principal: SOLO renderiza. No espera al servidor en ningún momento.
    while True:
        # Siempre tomar el frame más fresco (0 ms de lag de cámara)
        frame = cv2.resize(frame_actual.copy(), (640, 480))

        # Calcular FPS reales que vemos en pantalla
        tiempo_actual_fps = time.time()
        diff = tiempo_actual_fps - tiempo_previo_fps
        fps = (1 / diff) if diff > 0 else 0
        tiempo_previo_fps = tiempo_actual_fps

        # --- DIBUJAR INTERFAZ (UI) ---
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            estado_conexion,
            (380, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_conexion,
            2,
        )

        for det in detecciones_actuales:
            if det is None: continue
            top, right, bottom, left = det["location"]
            top, right, bottom, left = det.get("location", [0, 0, 0, 0])
            # Escalar coordenadas por 2 porque se envió como 320x240
            top, right, bottom, left = top*2, right*2, bottom*2, left*2

            nombre = det.get("identity", "Desconocido")
            tiene_uniforme = det.get("has_uniform", False)
            id_card = det.get("student_id")
            tiempo_actual = time.time()
            if tiene_uniforme == False and id_card and (tiempo_actual - ultimo_envio_alerta > intervalo_alerta):
                try:
                    _, buffer_img = cv2.imencode('.jpg', frame)
                    img_base64 = base64.b64encode(buffer_img).decode('utf-8')

                    # 🚀 2. Enviar la petición con la imagen incluida
                    payload = {
                        "idCard": id_card, 
                        "has_uniform": False,
                        "image": img_base64  # Enviamos el string Base64
                    }
                    requests.post(
                        url_alerts, 
                        json=payload,
                        timeout=2.0
                    )
                    ultimo_envio_alerta = tiempo_actual # Actualizamos el marcador de tiempo
                    print(f"🚨 Alerta enviada a Node para: {id_card}")
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            # Formatear el texto
            clothing_boxes = det.get("clothing_boxes", [])
            needs_full_body = det.get("needs_full_body_view", False)

            texto_uniforme = "UNIFORME: SI" if tiene_uniforme else "UNIFORME: NO"
            texto_final = f"{nombre} | {texto_uniforme}"

            if nombre == "Desconocido": # No coincide con estudiante
                color_recuadro = (0, 0, 255) # Rojo
            elif tiene_uniforme:        # Coincide y tiene uniforme
                color_recuadro = (0, 255, 0) # Verde
            else:                       # Coincide pero no tiene uniforme
                color_recuadro = (0, 165, 255) # Naranja

            cv2.rectangle(frame, (left, top), (right, bottom), color_recuadro, 2)
            cv2.putText(
                frame,
                texto_final,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color_recuadro,
                2,
            )

            for cb in clothing_boxes:
                c_name = cb.get("class", "Ropa")
                cx1, cy1, cx2, cy2 = cb.get("box", [0,0,0,0])
                # Escalar por 2
                cx1, cy1, cx2, cy2 = cx1*2, cy1*2, cx2*2, cy2*2
                
                c_valido = cb.get("valid", False)
                
                color_ropa = (0, 100, 0) if c_valido else (0, 0, 255)
                
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), color_ropa, 2)
                cv2.putText(
                    frame,
                    c_name.upper(),
                    (cx1, cy1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color_ropa,
                    2,
                )

            if needs_full_body:
                cv2.putText(
                    frame,
                    "ACERQUESE/ALEJESE PARA VER PANTALON",
                    (50, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255), 
                    2,
                )

        cv2.imshow("Totem AI Sentinel - En vivo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    hilo_sio = threading.Thread(target=conectar_sio_background, daemon=True)
    hilo_sio.start()
    open_cam()
