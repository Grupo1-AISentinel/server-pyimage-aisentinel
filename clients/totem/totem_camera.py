import cv2
import requests
import time

URL_SERVIDOR = "http://localhost:8000/api/detect"

def open_cam():
    cap = cv2.VideoCapture(0)
    ultimo_envio = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: continue

        cv2.imshow('Totem AI Sentinel - En vivo', frame)

        # Enviar 1 frame por segundo para no saturar el servidor ni la red
        if time.time() - ultimo_envio > 1.0:
            # Convertir el frame a formato JPG en memoria
            _, buffer = cv2.imencode('.jpg', frame)
            
            try:
                # Enviar al servidor FastAPI
                respuesta = requests.post(URL_SERVIDOR, files={"file": buffer.tobytes()})
                print("Detección:", respuesta.json())
            except requests.exceptions.ConnectionError:
                print("Esperando conexión con el servidor...")
                
            ultimo_envio = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    open_cam()