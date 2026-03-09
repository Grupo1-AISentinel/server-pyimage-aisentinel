import cv2
import requests
import time
import threading

URL_SERVIDOR = "http://localhost:8000/api/detect"

# Variables globales para UI fluida
detecciones_actuales = []
estado_conexion = "Esperando servidor..."
color_conexion = (0, 165, 255)  # Naranja por defecto


def enviar_frame_servidor(frame_bytes):
    global detecciones_actuales, estado_conexion, color_conexion
    try:
        respuesta = requests.post(URL_SERVIDOR, files={"file": frame_bytes})
        if respuesta.status_code == 200:
            datos = respuesta.json()
            detecciones_actuales = (
                datos if isinstance(datos, list) else datos.get("students", [])
            )
            estado_conexion = "Servidor Online"
            color_conexion = (0, 255, 0)  # Verde
    except requests.exceptions.ConnectionError:
        estado_conexion = "Sin conexion al servidor..."
        color_conexion = (0, 0, 255)  # Rojo
        detecciones_actuales = []  # Limpiar cuadros si se cae la red
        print("Esperando conexión con el servidor...")


def open_cam():
    cap = cv2.VideoCapture(0)  # Cambia a 0 para webcam real
    ultimo_envio = time.time()
    tiempo_previo_fps = 0

    while True:
        ret, frame_original = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame_original, (640, 480))

        # Calcular FPS
        tiempo_actual_fps = time.time()
        fps = (
            1 / (tiempo_actual_fps - tiempo_previo_fps) if tiempo_previo_fps > 0 else 0
        )
        tiempo_previo_fps = tiempo_actual_fps

        # --- DIBUJAR INTERFAZ (UI) ---

        # 1. FPS en la esquina superior izquierda
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        # 2. Estado del servidor en la esquina superior derecha
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
            top, right, bottom, left = det["location"]

            # Extraer los datos del JSON
            nombre = det.get("identity", "Desconocido")
            tiene_uniforme = det.get("has_uniform", False)

            # Formatear el texto
            texto_uniforme = "UNIFORME: SI" if tiene_uniforme else "UNIFORME: NO"
            texto_final = f"{nombre} | {texto_uniforme}"

            # Cambiar el color del recuadro: Verde (Con Uniforme) o Rojo (Sin Uniforme)
            color_recuadro = (0, 255, 0) if tiene_uniforme else (0, 0, 255)

            # Dibujar en pantalla
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

        cv2.imshow("Totem AI Sentinel - En vivo", frame)

        # Enviar 1 frame por segundo
        if time.time() - ultimo_envio > 1.0:
            _, buffer = cv2.imencode(".jpg", frame)
            hilo = threading.Thread(
                target=enviar_frame_servidor, args=(buffer.tobytes(),), daemon=True
            )
            hilo.start()
            ultimo_envio = time.time()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    open_cam()
