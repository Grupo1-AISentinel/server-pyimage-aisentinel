import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.clothing_engine import ClothingEngine

def populate_uniforms():
    print("Iniciando carga de catalogo de uniformes...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    images_promo33 = [
        os.path.join(base_dir, "img/uniforms/promo33/1.jpeg"),
        os.path.join(base_dir, "img/uniforms/promo33/2.jpeg"),
        os.path.join(base_dir, "img/uniforms/promo33/3.jpeg"),
        os.path.join(base_dir, "img/uniforms/promo33/4.jpeg")
    ]
    
    images_clasic = [
        os.path.join(base_dir, "img/uniforms/clasic/1.jpeg"),
        os.path.join(base_dir, "img/uniforms/clasic/2.jpeg"),
        os.path.join(base_dir, "img/uniforms/clasic/3.jpeg"),
        os.path.join(base_dir, "img/uniforms/clasic/4.jpeg"),
        os.path.join(base_dir, "img/uniforms/clasic/5.jpeg"),
        os.path.join(base_dir, "img/uniforms/clasic/6.jpeg")
    ]

    try:

        # Extraer y registrar Chumpas
        ClothingEngine.register_clothing_item(
            item_id="uniform_jacket_promo33", 
            item_type="jacket", 
            images_list=images_promo33
        )
        
        ClothingEngine.register_clothing_item(
            item_id="uniform_jacket_clasic", 
            item_type="jacket", 
            images_list=images_clasic
        )
        
        # Extraer y registrar Pantalones (Aprovechando que salen en las fotos de clasic)
        print("Extrayendo Pantalón de las fotos...")
        ClothingEngine.register_clothing_item(
            item_id="pantalon_oficial", 
            item_type="pants", 
            images_list=images_clasic
        )
        
        # Extraer y registrar Camisas (Aprovechando que salen en las fotos de clasic)
        print("Extrayendo Camisa de las fotos...")
        ClothingEngine.register_clothing_item(
            item_id="camisa_oficial", 
            item_type="shirt", 
            images_list=images_clasic
        )
    except Exception as e:
        print(f"[ERROR] Fallo al registrar el uniforme: {e}")

if __name__ == "__main__":
    populate_uniforms()