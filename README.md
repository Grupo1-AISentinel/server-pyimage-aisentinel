# AI Sentinel - PyImage Server

Microservicio de visión computacional basado en **FastAPI y Socket.io**. Encargado del reconocimiento facial, detección de asistencia y validación de uniformes mediante búsqueda vectorial.

## 📋 Requisitos Previos

Para este desarrollo utilizaremos **Docker**.

1. **Docker Desktop**: [Descargar e instalar aquí](https://www.docker.com/products/docker-desktop/).
   * *Asegúrate de que Docker esté abierto antes de ejecutar los comandos.*
2. **VS Code Extensions** (Recomendado):
   * Extension Pack de Python.
   * Extensión de Docker.

## Configuración inicial

### 1. Clonar el repositorio
```powershell
git clone [https://github.com/Grupo1-AISentinel/server-pyimage-aisentinel.git](https://github.com/Grupo1-AISentinel/
server-pyimage-aisentinel.git)
cd server-pyimage-aisentinel
```

### 2. Muevete a la rama develop y crea tu rama para la feature siguiendo el estándar: feature/carnet/tarea
```powershell
git checkout develop
git checkout -b feature/carnet/tarea
```
### 3. Coloca las variables de entorno .env
### 4. Levanta el contenedor de docker
```powershell
docker-compose up --build
```
## NOTA:
Si quieres evitar las advertencias de vscode con los paquetes. (Necesario tener python 10.0)
### 1. Crea un entorno virtual
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Intalar las dependencias básicas
```powershell
pip install fastapi uvicorn pydantic python-dotenv
```

### 3. Selecciona el interprete en vscode con ctrl + p -> Python: select interpreter
