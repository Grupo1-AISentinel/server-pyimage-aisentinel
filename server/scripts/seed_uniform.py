import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.clothing_engine import ClothingEngine

def populate_uniforms():
    print("Iniciando carga de catalogo de uniformes...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    images_promo33_close = [
        os.path.join(base_dir, "img/uniforms/promo33/close/1.jpeg"),
        os.path.join(base_dir, "img/uniforms/promo33/close/2.jpeg"),
        os.path.join(base_dir, "img/uniforms/promo33/close/3.jpeg")
    ]
    
    images_promo33_open = [
        os.path.join(base_dir, "img/uniforms/promo33/open/1.jpeg")
    ]
    

    images_clasic_close = [
        os.path.join(base_dir, "img/uniforms/clasic/close/1.jpeg"),
        os.path.join(base_dir, "img/uniforms/clasic/close/2.jpeg"),
        os.path.join(base_dir, "img/uniforms/clasic/close/3.jpeg"),
        os.path.join(base_dir, "img/uniforms/clasic/close/4.jpeg"),
        os.path.join(base_dir, "img/uniforms/clasic/close/5.jpeg"),
        os.path.join(base_dir, "img/uniforms/clasic/close/6.jpeg"),
        os.path.join(base_dir, "img/uniforms/clasic/close/7.jpeg")
    ]
    
    images_clasic_open = [
        os.path.join(base_dir, "img/uniforms/clasic/open/1.jpeg"),
        os.path.join(base_dir, "img/uniforms/clasic/open/2.jpeg"),
        os.path.join(base_dir, "img/uniforms/clasic/open/3.jpeg"),
        os.path.join(base_dir, "img/uniforms/clasic/open/4.jpeg"),
        os.path.join(base_dir, "img/uniforms/clasic/open/5.jpeg"),
        os.path.join(base_dir, "img/uniforms/clasic/open/6.jpeg"),
        os.path.join(base_dir, "img/uniforms/clasic/open/7.jpeg")
    ]

    images_tshirt = [
        os.path.join(base_dir, "img/uniforms/tshirt/1.jpeg"),
        os.path.join(base_dir, "img/uniforms/tshirt/2.jpeg"),
        os.path.join(base_dir, "img/uniforms/tshirt/3.jpeg"),
        os.path.join(base_dir, "img/uniforms/tshirt/4.jpeg"),
        os.path.join(base_dir, "img/uniforms/tshirt/5.jpeg"),
        os.path.join(base_dir, "img/uniforms/tshirt/6.jpeg")
    ]

    uniforms_to_register = [
        ("uniform_jacket_promo33_close", "jacket", images_promo33_close),
        ("uniform_jacket_promo33_open", "jacket", images_promo33_open),
        ("uniform_jacket_clasic_close", "jacket", images_clasic_close),
        ("uniform_jacket_clasic_open", "jacket", images_clasic_open),
        ("pantalon_oficial", "pants", images_clasic_close + images_clasic_open + images_tshirt + images_promo33_close + images_promo33_open),
        ("camisa_oficial", "shirt", images_tshirt)
    ]

    for item_id, item_type, img_list in uniforms_to_register:
        try:
            print(f"--- Procesando {item_id} ({item_type}) ---")
            ClothingEngine.register_clothing_item(item_id, item_type, img_list)
        except Exception as e:
            print(f"[ERROR EXCEPTION] Falló código en registrar {item_id}: {e}")

if __name__ == "__main__":
    populate_uniforms()