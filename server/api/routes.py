from fastapi import APIRouter
from api.schemas import StudentRegister

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "PYIMAGE Server is healthy", "version": "1.0.0"}

@router.post("/register")
def register_student(datos: StudentRegister):
    # Lógica para registrar al estudiante y procesar su imagen
    return {"message": f"Estudiante {datos.carnet} listo para procesar"}