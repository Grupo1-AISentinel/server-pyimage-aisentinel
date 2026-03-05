# AI Sentinel - Lado del Cliente (Cámaras)

Estos scripts representan los **Ojos** del sistema. Pueden correr en computadoras ligeras, Raspberry Pi (si el hardware lo permite) u ordenadores genéricos conectados a la red del servidor.

## ⚙️ Requisitos

A diferencia del servidor, los clientes **NO requieren Docker ni modelos pesados de Inteligencia Artificial**.
Necesitas instalar Python directamente en la máquina que ejecutará el cliente.

## 🚀 Instrucciones de Instalación

1.  Abre la terminal en esta carpeta (`clients`).
2.  Crea un entorno virtual (opcional pero muy recomendado):
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate   # En Windows
    ```
3.  Instala las dependencias Ligeras:
    ```powershell
    pip install -r requirements.txt
    ```
    _(Este archivo requirements.txt contiene únicamente `opencv-python` para abrir la cámara web y `requests` para mandar la foto por red)._

## 🎥 Ejecución del Tótem

El "Tótem" es la aplicación que graba video en vivo y manda fotogramas al servidor para validarlos:

```powershell
cd totem
python totem_camera.py
```

Asegúrate de que la variable `API_URL` dentro del script esté apuntando correctamente a la IP de la computadora que tiene el Servidor Docker (Por ejemplo `http://192.168.1.10:8000/api/detect` en vez de `localhost` si se ejecutan en máquinas distintas).

## 🛠️ Solución de Problemas Comunes

- **"Connection Refused" (Conexión rechazada):** El Tótem no puede ver al servidor. Si el servidor y el cliente están en computadoras separadas, cambia `localhost` por la IP local del servidor (IPv4).
- **Cámara no detectada:** Revisa en el código `cv2.VideoCapture(0)`. Si tienes varias cámaras conectadas, cambia el `0` por `1` o `2`.
