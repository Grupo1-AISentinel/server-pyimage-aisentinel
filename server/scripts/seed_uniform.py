import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.clothing_engine import ClothingEngine

def populate_uniforms():
    print("Iniciando carga de catalogo de uniformes...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    images_jacket = [
        os.path.join(base_dir, "img/students/asiliezar/1.jpeg"),
        os.path.join(base_dir, "img/students/asiliezar/2.jpeg"),
        os.path.join(base_dir, "img/students/asiliezar/3.jpeg")
    ]

    try:

        ClothingEngine.register_uniform(
            uniform_id="uniform_jacket", 
            uniform_type="jacket promo 33", 
            images_list=images_jacket
        )
    except Exception as e:
        print(f"[ERROR] Fallo al registrar el uniforme: {e}")

if __name__ == "__main__":
    populate_uniforms()