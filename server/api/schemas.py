from pydantic import BaseModel

class StudentRegister(BaseModel):
    carnet: str
    nombre: str = "Sin nombre"