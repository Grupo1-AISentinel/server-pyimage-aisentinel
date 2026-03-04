import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.biometric_engine import BiometricEngine

def poblar_estudiantes():
    print("Iniciando carga de estudiantes...")
    
    students_to_seed = [
        {"card": "2024342", "name": "Angel Siliezar", "images": ["img/students/asiliezar/1.jpeg", "img/students/asiliezar/2.jpeg", "img/students/asiliezar/3.jpeg"]},
        {"card": "2024295", "name": "Isaac Tiguila", "images": ["img/students/itiguila/1.jpeg"]},
        {"card": "2024442", "name": "Jeferson Yaxon", "images": ["img/students/jyaxon/1.jpeg"]},
        {"card": "2024449", "name": "Anderson Sosa", "images": ["img/students/asosa/asosa.jpeg"]},
        {"card": "2024392", "name": "Wilson Florian", "images": ["img/students/wflorian/1.jpeg"]},
        {"card": "2021211", "name": "Angel Lucero", "images": ["img/students/alucero/1.jpeg"]}
    ]

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    engine = BiometricEngine()

    for student in students_to_seed:
        # Convertir rutas relativas a rutas absolutas
        rutas_absolutas = [os.path.join(base_dir, img) for img in student["images"]]
        
        try:
            # Hacer el promedio
            engine.register_face(student["card"], student["name"], rutas_absolutas)
        except Exception as e:
            print(f"Error con {student['name']}: {e}")

    print("EXITO: Carga de estudiantes completada.")

if __name__ == "__main__":
    poblar_estudiantes()