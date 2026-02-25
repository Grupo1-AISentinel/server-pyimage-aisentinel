# AI Sentinel

Sistema de visión computacional dividido en una arquitectura Cliente-Servidor para el control de asistencia y seguridad escolar.

## Estructura del Proyecto

El proyecto está dividido en dos grandes áreas:

* **`/server` (El Cerebro):** Contiene la API, la Base de Datos Vectorial (ChromaDB) y la lógica de Inteligencia Artificial. Corre sobre Docker.
* **`/clients` (Los Ojos):** Scripts de Python independientes que ejecutan la cámara, capturan fotos y las envían al servidor.

---

## Lado del SERVIDOR (Backend)

Este componente debe correr siempre en la computadora principal o servidor.

### Requisitos
* Docker Desktop instalado y corriendo.
* Archivo `.env` configurado en la raíz.

### Puesta en Marcha
1.  Ubícate en la raíz del proyecto (donde está el `docker-compose.yml`).
2.  Levanta el servicio:
    ```powershell
    docker-compose up --build
    ```
3.  El servidor estará escuchando en: `http://localhost:8000`

---

## 📷 2. Lado del CLIENTE (Cámaras y Registro)

Estos scripts pueden correr en cualquier computadora conectada a la misma red que el servidor (o la misma máquina).

### Requisitos del Cliente
Necesitas Python instalado en tu máquina (no Docker).

1.  Ve a la carpeta de clientes:
    ```powershell
    cd clients/totem
    ```
2.  Crea un entorno virtual (opcional pero recomendado):
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  Instala las dependencias LIGERAS del cliente:
    ```powershell
    pip install -r requirements.txt
    ```
    *(Nota: Este requirements.txt solo debe tener `opencv-python` y `requests`).*

### Ejecución

* **Para el Tótem de Entrada (Reconocimiento en Vivo):**
    ```powershell
    python totem_camera.py
    ```

* **Para Registrar Nuevos Alumnos (Herramienta Administrativa):**
    ```powershell
    cd ../admin_tools
    python register.py
    ```

---

## 🔧 Desarrollo y Contribución

### Git Flow
1.  **Rama Principal:** `develop`
2.  **Nueva Funcionalidad:** Crea tu rama `feature/carnet/nombre-tarea`.

### Solución de Problemas Comunes
* **Error de conexión en el cliente:** Verifica que `API_URL` en los scripts de Python apunte a la IP correcta de tu servidor Docker (ej. `localhost` o `192.168.x.x`).
* **Docker no levanta:** Asegúrate de que el puerto 8000 no esté ocupado.