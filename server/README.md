# AI Sentinel - Lado del Servidor (Backend AI)

Este componente es el **Cerebro** del sistema. Recibe imágenes, localiza estudiantes, analiza los rostros y verifica estrictamente el uso completo del uniforme.

## ⚙️ Requisitos

- Docker Desktop instalado y corriendo.
- Archivo `.env` configurado.

## 🚀 Puesta en Marcha

1. Ubícate en la raíz del proyecto (una carpeta atrás).
2. Construye y levanta el contenedor en segundo plano:
   ```powershell
   docker-compose up --build -d
   ```
3. La API estará escuchando peticiones en: `http://localhost:8000`

---

## 🧠 Modelos de Inteligencia Artificial

El sistema utiliza 3 modelos en conjunto para lograr **Cero Falsos Positivos**:

1. **`yolo26n.pt` (El Localizador de Personas):**
   Modelo nano de Ultralytics YOLO26 para detectar **la silueta de las personas** (los cuerpos enteros). Optimizado para recursos limitados (ej. GTX 1050 Ti). Busca en `server/` y, si no existe, descarga automáticamente vía Ultralytics.
2. **`AdamCodd/YOLOv11n-face-detection` (Detección de Rostros):**
   YOLO especializado en caras, alojado en Hugging Face. Detecta las ubicaciones (bounding boxes) de rostros. Luego `face_recognition` (dlib) extrae embeddings 128D de esas regiones y ChromaDB compara con los registrados. Pipeline: **YOLO detecta dónde** → **dlib identifica quién**.
3. **`best.pt` (El Localizador de Ropa Custom):**
   Es tu modelo personalizado (ubicado canónicamente en `server/best.pt`) reentrenado sobre YOLO26s. YOLO detecta qué prendas lleva el alumno usando clases estructurales como `jacket_open`, `jacket_close`, `shirt`, `pant` y `accesory`. El `Clothing Engine` valida luego la oficialidad de la prenda con embeddings y referencias de color almacenadas en ChromaDB.

## 🧥 Reglas de Uniforme

- `jacket_close` válida: no exige camisa oficial.
- `jacket_open` válida: sí exige camisa oficial.
- `shirt` válida sin chumpa: se considera parte superior válida.
- `pants` válido: siempre obligatorio.
- `accesory` detectado: invalida el uniforme.

La detección estructural distingue `jacket_open` y `jacket_close`, pero el catálogo vectorial sigue usando una sola familia lógica `jacket` para comparar oficialidad.

## 📦 Herramientas Administrativas

Para poblar la base de datos con caras y ropa (Semillas):

```powershell
# Opción 1: Ejecutar todo de una vez (recomendado)
python scripts/seed_all.py

# Opción 2: Ejecutar por separado (orden importante)
python scripts/seed_students.py   # Pobla caras
python scripts/seed_uniform.py    # Resetea uniformes (según RESET_UNIFORM_COLLECTION) y pobla catálogo
```

**Limpieza de uniformes:** Por defecto `seed_uniform` resetea la colección de uniformes para evitar mezclar embeddings antiguos con los nuevos.
- Para conservar la colección de uniformes existente: `RESET_UNIFORM_COLLECTION=false python scripts/seed_uniform.py`

**Modelos externos:** Los modelos YOLO (face, persona, ResNet) se descargan automáticamente la primera vez. `YOLO_CONFIG_DIR=/tmp/Ultralytics` evita el warning si el directorio por defecto no es escribible.

_(Ejecutar desde la carpeta `server/` o ajustar rutas si corres desde la raíz del proyecto)._
