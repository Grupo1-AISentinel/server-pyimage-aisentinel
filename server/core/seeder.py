import os
import logging
from services.biometric_engine import BiometricEngine
from api.models.Student import StudentModel
from db.connection import client

logger = logging.getLogger(__name__)


def seed_users():
    """
    Cargar usuarios, si no existen se salta este paso.
    """
    logger.info("Iniciando Data Seeder...")

    # Datos para enviar al motor - image_path list
    students_to_seed = [
        StudentModel(
            carnet="2024342", name="Angel Siliezar", image_path=["img/students/asiliezar/1.jpeg", "img/students/asiliezar/2.jpeg", "img/students/asiliezar/3.jpg", "img/students/asiliezar/4.jpeg", "img/students/asiliezar/5.jpeg"]
        ),
        StudentModel(
            carnet="2024295", name="Isaac Tiguila", image_path=["img/students/itiguila/1.jpeg"]
        ),
        StudentModel(
            carnet="2024442", name="Jeferson Yaxon", image_path=["img/students/jyaxon/1.jpeg"]
        ),
        StudentModel(
            carnet="2024449", name="Anderson Sosa", image_path=["img/students/asosa/1.jpeg"]
        ),
        StudentModel(
            carnet="2024392", name="Wilson Florian", image_path=["img/students/wflorian/1.jpeg"]
        ),
        StudentModel(
            carnet="2021211", name="Angel Lucero", image_path=["img/students/alucero/1.jpeg"]
        )
    ]
    
    for student in students_to_seed:
        # Verificar si ya existe
        try:
            collection = client.get_collection("student_faces")
            # Buscar si ya existe
            existing = collection.get(ids=[student.carnet])

            if existing and existing["ids"]:
                logger.info(f"El usuario {student.name} ({student.carnet}) ya existe. Saltando registro.")
                continue
        except Exception:
            logger.warning(f"No se pudo verificar la existencia de {student.name} en la DB. Intentando registrar de todos modos...")

        # Registrar a cada usuario
        try:
            logger.info(f"Registrando usuario: {student.name}...")
            BiometricEngine.register_face(student.carnet, student.name, student.image_path)
            logger.info(f"SEEDER EXITOSO: {student.name} ha sido registrado en la DB.")
        except Exception as e:
            logger.error(f"Error en el seeder: {e}")
    


if __name__ == "__main__":
    seed_users()
