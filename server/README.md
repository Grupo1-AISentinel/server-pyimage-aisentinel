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

1. **`yolov8n.pt` (El Localizador de Personas):**
   Este modelo general de YOLO se descarga automáticamente al iniciar el servidor. Su única función es detectar **la silueta de las personas** (los cuerpos enteros). Esto permite que el sistema analice a los estudiantes como "Individuos Completos" y cruce la información del rostro con la ropa correcta.
2. **`AdamCodd/YOLOv11n-face-detection` (Reconocimiento Facial):**
   Modelo _State-of-the-Art_ alojado en Hugging Face. Detecta los rostros incluso bajo condiciones difíciles (de lado, borrosos o lejanos) y envía las coordenadas a nuestro motor biométrico, eliminando la detección errónea de rostros en paredes o fondos. (Se descarga automáticamente vía internet).
3. **`best.pt` (El Localizador de Ropa Custom):**
   Es tu modelo personalizado (ubicado en `server/best.pt`). YOLO detecta qué prendas (Coat, clothes top, Pants) lleva el alumno usando las cajas. Si detecta la ropa necesaria, el "Clothing Engine" tomará un recorte exacto de la chumpa y comparará su código de colores/diseño (Vector) con los uniformes oficiales guardados en ChromaDB, dando el veredicto final.

## 📦 Herramientas Administrativas

Para poblar la base de datos con caras y ropa (Semillas):

```powershell
# Ejecutar estos scripts pobladores una vez levantado el servidor
python scripts/seed_students.py
python scripts/seed_uniform.py
```

_(Asegúrate de ejecutar esto desde esta carpeta o la carpeta raíz ajustando las rutas)._
