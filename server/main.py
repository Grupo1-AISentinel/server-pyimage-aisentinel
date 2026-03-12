from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from scripts.seed_students import STUDENTS_MAP
import uvicorn
import os
import mimetypes
import logging
import traceback
import base64
import io
import time
import cv2
import requests 
import numpy as np
import socketio
import asyncio
from threading import Thread
from PIL import Image, ImageOps


# Importar las rutas desde la carpeta api
from api.routes import router as api_router 

try:
    from services.biometric_engine import BiometricEngine
    logger_name = "AISentinel"
except ImportError:
    class BiometricEngine:
        def register_face(self, carnet, nombre, fotos):
            return False
    logger_name = "AISentinel-Debug"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(logger_name)
sio = socketio.Client(reconnection=True, reconnection_attempts=10, reconnection_delay=5)
engine = BiometricEngine()
app = FastAPI(title="AISentinel")

# Nuevo servidor Socket.IO para el Totem (Cliente cámara)
sio_server = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*', max_http_buffer_size=10485760)
socket_app = socketio.ASGIApp(sio_server, other_asgi_app=app)

# Semáforos por sid para cada nivel de detección.
# Ambos descartan frames si ya hay uno en procesamiento → evitan cola de frames obsoletos.
_semaforos_sid: dict[str, asyncio.Semaphore] = {}      # Pipeline completo (lento)
_semaforos_rapido: dict[str, asyncio.Semaphore] = {}   # Solo YOLO face (rápido)

@sio_server.event
async def connect(sid, environ, auth):
    _semaforos_sid[sid] = asyncio.Semaphore(1)
    _semaforos_rapido[sid] = asyncio.Semaphore(1)
    logger.info(f"📹 Cámara Totem conectada a Socket.IO: {sid}")

@sio_server.event
async def disconnect(sid):
    _semaforos_sid.pop(sid, None)
    _semaforos_rapido.pop(sid, None)
    logger.info(f"🔌 Cámara Totem desconectada de Socket.IO: {sid}")

@sio_server.event
async def detect_boxes(sid, data):
    """
    Detección rápida: solo YOLO face sin face_recognition encoding.
    Objetivo: respuesta en ~50-80ms para que las cajas sigan al instante a la persona.
    """
    sem = _semaforos_rapido.get(sid)
    if sem is None or sem.locked():
        return

    async with sem:
        try:
            img_bytes = data.get('image')
            if not img_bytes:
                return
            from api.routes import process_frame_fast_from_bytes
            resultado = await asyncio.to_thread(process_frame_fast_from_bytes, img_bytes)
            if resultado is not None:
                await sio_server.emit('detect_boxes_result', resultado, to=sid)
        except Exception as e:
            logger.error(f"Error en detect_boxes: {e}")

@sio_server.event
async def detect_frame(sid, data):
    """
    Detección completa: identidad + uniforme. Más lento (~300-600ms en CPU).
    El cliente lo llama cada ~2s. Se descarta si el servidor ya está procesando.
    """
    sem = _semaforos_sid.get(sid)
    if sem is None or sem.locked():
        return

    async with sem:
        try:
            if 'image' not in data:
                return

            img_bytes = data['image']
            if img_bytes:
                from api.routes import process_frame_from_bytes
                resultados_finales = await asyncio.to_thread(process_frame_from_bytes, img_bytes, engine)

                if resultados_finales is not None:
                    await sio_server.emit('detect_results', resultados_finales, to=sid)
        except Exception as e:
            logger.error(f"Error procesando frame de camara vía Socket.IO: {e}\n{traceback.format_exc()}")

# Configuración estricta de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@sio.event
def connect():
    logger.info("✅ Conectado bidireccionalmente con Node.js (Puerto 3067)")

@sio.event
def disconnect():
    logger.info("❌ Desconectado de Node.js")

@sio.on('enviar_a_python')
def handle_node_data(data):
    carnet = data.get('carnet', 'Desconocido')
    nombre = data.get('nombre', 'Sin Nombre')
    fotos_raw = data.get('fotos', [])

    try:
        logger.info(f"📸 Procesando registro para: {nombre} ({carnet})")
        lista_imagenes = []

        for f in fotos_raw:
            try:
                raw_data = f.get('buffer') or f.get('base64')
                if not raw_data:
                    continue

                if isinstance(raw_data, str):
                    if "," in raw_data:
                        raw_data = raw_data.split(",")[1]
                    img_bytes = base64.b64decode(raw_data)
                else:
                    img_bytes = raw_data

                image = Image.open(io.BytesIO(img_bytes))
                image = ImageOps.exif_transpose(image)  # <--- Corrige la rotación automática
                image = image.convert('RGB')
                
                img_np = np.array(image)

                lista_imagenes.append(img_np)
                logger.info(f"   - Imagen decodificada: {image.size[0]}x{image.size[1]}")

            except Exception as e:
                logger.error(f"   - Error procesando imagen individual: {e}")

        if not lista_imagenes:
            raise ValueError("No se pudieron decodificar imágenes válidas.")

        success = engine.register_face_from_socket(carnet, nombre, lista_imagenes)

        if success:
            logger.info(f"🎯 Registro biométrico EXITOSO: {carnet}")
            sio.emit('python_registro_completado', {'carnet': carnet, 'status': 'success'})
        else:
            logger.error(f"❌ La IA no detectó rostros en las imágenes de {carnet}")
            sio.emit('python_registro_completado', {
                'carnet': carnet, 
                'status': 'error', 
                'message': 'No se detectó un rostro claro. Intente de nuevo.'
            })

    except Exception as e:
        logger.error(f"❌ Error crítico en el flujo: {e}")
        sio.emit('python_registro_completado', {'carnet': carnet, 'status': 'error', 'message': str(e)})

@sio.on('enviar_uniforme_a_python')
def handle_uniform_data(data):
    """
    Recibe datos de uniforme desde Node.js vía Socket.IO bidireccional.
    Campos esperados:
      - name: str (nombre del uniforme, ej: "Chumpa Clásica")
      - type: str (JACKET | TSHIRT | PANTS)
      - fotos: list[{buffer|base64}] (imágenes del uniforme)
    """
    from services.clothing_engine import ClothingEngine

    uniform_name = data.get('name', 'Sin Nombre')
    uniform_type = data.get('type', '')
    fotos_raw = data.get('fotos', [])

    try:
        logger.info(f"👔 Procesando registro de uniforme: {uniform_name} (tipo: {uniform_type})")
        lista_imagenes = []

        for f in fotos_raw:
            try:
                raw_data = f.get('buffer') or f.get('base64')
                if not raw_data:
                    continue

                if isinstance(raw_data, str):
                    if "," in raw_data:
                        raw_data = raw_data.split(",")[1]
                    img_bytes = base64.b64decode(raw_data)
                else:
                    img_bytes = raw_data

                image = Image.open(io.BytesIO(img_bytes))
                image = ImageOps.exif_transpose(image)
                image = image.convert('RGB')

                img_np = np.array(image)
                lista_imagenes.append(img_np)
                logger.info(f"   - Imagen uniforme decodificada: {image.size[0]}x{image.size[1]}")

            except Exception as e:
                logger.error(f"   - Error procesando imagen de uniforme: {e}")

        if not lista_imagenes:
            raise ValueError("No se pudieron decodificar imágenes válidas del uniforme.")

        # Generar un ID único basado en el nombre (slug limpio)
        item_id = uniform_name.lower().replace(" ", "_").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")

        success, count = ClothingEngine.register_clothing_from_numpy(
            item_id=item_id,
            item_type=uniform_type,
            images_np=lista_imagenes,
            extra_meta={"nombre_display": uniform_name}
        )

        if success:
            logger.info(f"🎯 Registro de uniforme EXITOSO: {uniform_name} → {count} vectores")
            sio.emit('python_uniforme_registrado', {
                'name': uniform_name,
                'type': uniform_type,
                'status': 'success',
                'vectors': count
            })
        else:
            logger.error(f"❌ No se detectaron prendas '{uniform_type}' en las imágenes de {uniform_name}")
            sio.emit('python_uniforme_registrado', {
                'name': uniform_name,
                'type': uniform_type,
                'status': 'error',
                'message': f'No se detectó prenda tipo {uniform_type} en las imágenes. Intente con fotos más claras.'
            })

    except Exception as e:
        logger.error(f"❌ Error crítico registrando uniforme: {e}")
        sio.emit('python_uniforme_registrado', {
            'name': uniform_name,
            'type': uniform_type,
            'status': 'error',
            'message': str(e)
        })

def start_socket():
    node_url = "http://host.docker.internal:3067"
    while True:
        if not sio.connected:
            try:
                logger.info(f"⏳ Intentando conectar a {node_url}...")
                sio.connect(node_url)
                sio.wait()
            except Exception:
                time.sleep(5)
        else:
            time.sleep(10)

app.include_router(api_router)

def auto_sync_with_node():
    # CAMBIO CLAVE: Usar host.docker.internal para salir del contenedor
    URL_NODE = "http://host.docker.internal:3067/AISentinelAdmin/v1/students/auto-sync" 
    print(f"\n[AUTO-SYNC] Intentando sincronizar con: {URL_NODE}")
    payload = []
    for key, (carnet, nombre_completo) in STUDENTS_MAP.items():
        nombres = nombre_completo.split(" ", 1)
        payload.append({
            "idCard": carnet,
            "studentName": nombres[0],
            "studentSurname": nombres[1] if len(nombres) > 1 else "",
            "email": f"{key}-{carnet}@kinal.edu.gt",
            "grade": "6TO" 
        })
    try:
        logger.info(f"[AUTO-SYNC] Enviando {len(payload)} estudiantes...")
        print(f"[AUTO-SYNC] Enviando {len(payload)} estudiantes...")
        response = requests.post(URL_NODE, json=payload, timeout=10)
        if response.status_code == 200:
            res_data = response.json()
            print(f"✅ [AUTO-SYNC] Éxito. Creados: {res_data.get('newlyCreated')}")
            logger.info(f"[AUTO-SYNC] Éxito. Creados: {res_data.get('newlyCreated')}")
        else:
            print(f"⚠️ [AUTO-SYNC] Node respondió error {response.status_code}: {response.text}")
            logger.error(f"[AUTO-SYNC] Node respondió error {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"❌ [AUTO-SYNC] Error: No se pudo conectar a Node. ¿Está el servidor en el puerto 3067?")
        logger.error(f"[AUTO-SYNC] Error: No se pudo conectar a Node. ¿Está el servidor en el puerto 3067?")
    except Exception as e:
        print(f"❌ [AUTO-SYNC] Error inesperado: {e}")
        logger.error(f"[AUTO-SYNC] Error inesperado: {e}")

def auto_sync_uniforms_with_node():
    """Sincroniza los uniformes con Node.js enviando nombre, tipo y thumbnail recortado por YOLO."""
    URL_NODE = "http://host.docker.internal:3067/AISentinelAdmin/v1/uniforms/auto-sync"
    from services.clothing_engine import ClothingEngine
    
    print(f"\n[AUTO-SYNC-UNIFORMS] Intentando sincronizar con: {URL_NODE}")

    VALID_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    uniforms_dir = os.path.join(base_dir, "img", "uniforms")

    # Mapa: (tipo_interno) -> (nombre para Node, tipo para Node)
    # Se busca en carpetas específicas, pero pants puede buscar en cualquier lado.
    UNIFORMS_CONFIG = [
        {"search_dirs": ["clasic/close", "clasic/open"], "name": "Chumpa Clásica",   "type": "JACKET", "internal": "jacket"},
        {"search_dirs": ["tshirt"],                     "name": "Camisa Oficial",   "type": "TSHIRT", "internal": "shirt"},
        {"search_dirs": ["pants", "clasic/close", "tshirt"], "name": "Pantalon Oficial", "type": "PANTS",  "internal": "pants"},
    ]

    payload = []

    for cfg in UNIFORMS_CONFIG:
        uniform_name = cfg["name"]
        uniform_type = cfg["type"]
        internal_key = cfg["internal"]
        
        found_crop_b64 = None
        mime_type = "image/jpeg"

        # Buscar en las carpetas candidatas hasta encontrar un recorte válido por YOLO
        for rel_path in cfg["search_dirs"]:
            dir_path = os.path.join(uniforms_dir, rel_path)
            if not os.path.isdir(dir_path):
                continue
            
            images = sorted([
                f for f in os.listdir(dir_path)
                if os.path.splitext(f.lower())[1] in VALID_EXT
            ])

            for img_name in images:
                img_path = os.path.join(dir_path, img_name)
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                
                # Usar el motor de ropa para extraer recortes
                detections = ClothingEngine.extract_all_clothing_from_image(frame, conf=0.30)
                crops = detections.get(internal_key, [])
                
                if crops:
                    # Tomar el primer recorte encontrado
                    crop = crops[0]
                    # Convertir el recorte (numpy) a bytes JPG
                    success, buffer = cv2.imencode(".jpg", crop)
                    if success:
                        found_crop_b64 = base64.b64encode(buffer).decode("utf-8")
                        logger.info(f"[AUTO-SYNC-UNIFORMS] Recorte generado para {uniform_name} usando {img_name}")
                        break # Encontrado en esta imagen
            
            if found_crop_b64:
                break # Encontrado para este uniforme

        if not found_crop_b64:
            logger.warning(f"[AUTO-SYNC-UNIFORMS] No se pudo encontrar un recorte de YOLO para: {uniform_name}")
            continue

        payload.append({
            "name": uniform_name,
            "type": uniform_type,
            "thumbnail": {
                "data": found_crop_b64,
                "mimetype": mime_type
            }
        })

    if not payload:
        logger.warning("[AUTO-SYNC-UNIFORMS] No se encontraron uniformes para sincronizar.")
        return

    created = 0
    for uniform in payload:
        try:
            logger.info(f"[AUTO-SYNC-UNIFORMS] Enviando: {uniform['name']} ({uniform['type']})...")
            response = requests.post(URL_NODE, json=uniform, timeout=15)
            if response.status_code == 200 or response.status_code == 201:
                created += 1
                print(f"✅ [AUTO-SYNC-UNIFORMS] {uniform['name']} — OK")
                logger.info(f"[AUTO-SYNC-UNIFORMS] {uniform['name']} — OK")
            else:
                print(f"⚠️ [AUTO-SYNC-UNIFORMS] {uniform['name']} — Error {response.status_code}: {response.text}")
                logger.error(f"[AUTO-SYNC-UNIFORMS] {uniform['name']} — Error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            print(f"❌ [AUTO-SYNC-UNIFORMS] Error: No se pudo conectar a Node.")
            logger.error(f"[AUTO-SYNC-UNIFORMS] Error: No se pudo conectar a Node.")
            break
        except Exception as e:
            print(f"❌ [AUTO-SYNC-UNIFORMS] Error inesperado con {uniform['name']}: {e}")
            logger.error(f"[AUTO-SYNC-UNIFORMS] Error inesperado: {e}")

    print(f"[AUTO-SYNC-UNIFORMS] Completado: {created}/{len(payload)} uniformes sincronizados.")

@app.on_event("startup")
async def startup_event():
    logger.info("🟢 Servidor AI Sentinel Iniciado...")
    Thread(target=auto_sync_with_node, daemon=True).start()
    Thread(target=auto_sync_uniforms_with_node, daemon=True).start()
    import torch
    if torch.cuda.is_available():
        gpu_name  = torch.cuda.get_device_name(0)
        vram_gb   = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        vram_free = torch.cuda.mem_get_info(0)[0] / 1024 ** 3
        logger.info(f"[CUDA] ✅ GPU NVIDIA disponible: {gpu_name}")
        logger.info(f"[CUDA] VRAM total: {vram_gb:.1f} GB | VRAM libre: {vram_free:.1f} GB")
        logger.info(f"[CUDA] Todos los modelos (YOLO-Face, YOLO-Person, YOLO-Clothing, ResNet) usarán GPU")
    else:
        logger.warning("[CUDA] ⚠️  Sin GPU CUDA — el servidor usará CPU para toda inferencia (rendimiento reducido)")

    thread = Thread(target=start_socket, daemon=True)
    thread.start()
    

@app.get("/")
def read_root():
    return {"status": "online", "service": "AISentinel Facial Recognition API"}

if __name__ == "__main__":
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)