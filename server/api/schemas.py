from pydantic import BaseModel
from typing import Optional, Any

class StudentRegister(BaseModel):
    card: str
    name: str = "Sin nombre"
    images: list = None

class DetectResponse(BaseModel):
    
    status: str
    student: Optional[Any] = None
    has_uniform: Optional[bool] = None
    clothing_distance: Optional[float] = None