from ast import List

from pydantic import BaseModel
from typing import Optional, List, Any


class StudentRegister(BaseModel):
    card: str
    name: str = "Sin nombre"
    images: list = None


class DetectedStudent(BaseModel):
    location: Any
    identity: str
    student_id: Optional[str] = None
    color: Any
    confidence: str
    has_uniform: bool = False
    clothing_distance: float = 999.0


class DetectResponse(BaseModel):
    status: str
    students: List[DetectedStudent] = []
