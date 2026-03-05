# AI Sentinel - Arquitectura General

Sistema de visión computacional dividido en una arquitectura Cliente-Servidor para el control de asistencia y seguridad escolar con validación estricta de uniformes mediante Inteligencia Artificial (YOLO11 + ResNet18).

## 🏗️ Estructura del Proyecto

El proyecto está dividido en dos grandes áreas, cada una con su propia documentación detallada:

- **[`/server` (El Cerebro)](./server/README.md):** Contiene la API, la Base de Datos Vectorial (ChromaDB) y la lógica Híbrida de Inteligencia Artificial (YOLO + ResNet). Corre sobre Docker.
- **[`/clients` (Los Ojos)](./clients/README.md):** Scripts de Python independientes que ejecutan la cámara, capturan fotos en vivo del tótem y las envían al servidor para su procesamiento.

---

## 📋 Requisitos Generales del Sistema

1. **Hardware del Servidor:** PC con capacidad para correr modelos YOLO (idealmente con GPU NVIDIA, pero soporta CPU).
2. **Software del Servidor:** Docker y Docker Compose instalados.
3. **Hardware de Clientes:** Cámaras web conectadas a computadoras (Tótems).
4. **Red:** Los clientes deben estar en la misma red local (LAN) o tener acceso a la IP del servidor.

---

## 🚀 Guía Rápida de Ejecución

Esta es una vista general. Para detalles, consulta el README de cada carpeta específica.

### 1. Levantar el Servidor

```powershell
# En la raíz del proyecto
docker-compose up --build -d
```

El servidor de Inteligencia Artificial quedará operando en `http://localhost:8000`.

### 2. Levantar la Cámara Cliente

```powershell
cd clients/totem
python totem_camera.py
```

---

## 💡 Recomendaciones del Sistema Híbrido AI

Hemos alcanzado **Cero Falsos Positivos** usando una arquitectura híbrida:

- **Detección Estructural (YOLO):** Ubica _dónde_ está la ropa de interés (Ej. Chumpas, pantalones).
- **Validación de Identidad (ResNet18):** Recorta la prenda usando el cuadro que dio YOLO y usa ResNet para confirmar si el diseño/color coincide con los uniformes de la Base de Datos.

**Para mantener el sistema dinámico:**
No re-entrenes a YOLO cada vez que cambie la chumpa de promoción. Utiliza el endpoint del servidor `/register/uniform` para inyectar a la base de datos la foto de la "nueva chumpa promocional".

---

## 🔧 Git Flow y Contribución

1.  **Rama Principal:** `develop`
2.  **Nueva Funcionalidad:** Crea tu rama `feature/carnet/nombre-tarea`.
