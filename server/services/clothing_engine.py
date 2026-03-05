import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from ultralytics import YOLO
import cv2
import numpy as np

from db.cruds.crud_uniform import save_uniform_vector, search_uniform_by_vector

# Carga de recursos en Memoria
DEVICE = '0' if torch.cuda.is_available() else 'cpu'

# 1. Modelo YOLO General: Detectar a las personas
yolo_person_model = YOLO('yolov8n.pt') 

# 2. Modelo YOLO Estructural (Genérico): Detectar dónde está la ropa
# Clases del modelo best.pt: 0: 'Coat', 1: 'Pants', 2: 'Shoes', 3: 'Skirt', 4: 'clothes top', 5: 'padded jacket'
yolo_clothing_structure_model = YOLO('best.pt')

# 3. Modelo Vectorial (Identidad/Color de la ropa)
resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
resnet_model.eval()
resnet_model = resnet_model.to(DEVICE)

class ClothingEngine:

    @staticmethod
    def extract_all_torsos(frame):
        """Detecta a todas las personas en un frame y extrae su recorte (cuerpo) para análisis posterior."""
        results = yolo_person_model(frame, classes=[0], device=DEVICE, verbose=False)
        personas = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size != 0:
                    personas.append({
                        "box": (x1, y1, x2, y2),
                        "crop": crop
                    })
        return personas

    @staticmethod
    def _get_embedding(cropped_image_bgr):
        """Convierte un recorte de ropa exacto (ej. solo la chumpa) a un vector matemático de 512D."""
        if cropped_image_bgr is None or cropped_image_bgr.size == 0:
            return None
            
        color_rgb = cv2.cvtColor(cropped_image_bgr, cv2.COLOR_BGR2RGB)
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        img_tensor = transform(color_rgb).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            vector = resnet_model(img_tensor).flatten().tolist()
        return vector

    @staticmethod
    def register_clothing_item(item_id: str, item_type: str, images_list: list):
        """
        [ADMIN API] - Registra una prenda (ej. chumpa promo 2026).
        Busca la prenda con YOLO 'best.pt', la recorta, saca el vector y lo guarda en ChromaDB.
        """
        vectors_buffer = []
        
        # Mapear item_type interno a las clases que el modelo 'best.pt' entiende
        valid_model_classes = []
        if item_type == 'jacket':
            valid_model_classes = ['Coat', 'padded jacket']
        elif item_type == 'shirt':
            valid_model_classes = ['clothes top']
        elif item_type == 'pants':
            valid_model_classes = ['Pants']

        for image_path in images_list:
            frame = cv2.imread(image_path)
            if frame is None:
                continue

            results = yolo_clothing_structure_model(frame, device=DEVICE, verbose=False)
            
            clothing_crop = None
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    # Validar si el objeto detectado corresponde al tipo que queremos registrar
                    if class_name in valid_model_classes: 
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        clothing_crop = frame[y1:y2, x1:x2]
                        break # Tomamos la primera coincidencia
                if clothing_crop is not None:
                     break
                     
            if clothing_crop is not None:
                vector = ClothingEngine._get_embedding(clothing_crop)
                vectors_buffer.append(vector)
                
        if not vectors_buffer:
            raise ValueError(f"[ERROR] YOLO no detectó un/a '{item_type}' claro en las imágenes para registrarlo.")
            
        master_vector = np.mean(vectors_buffer, axis=0).tolist()
        metadata = {"tipo": item_type, "valido": True}
        save_uniform_vector(item_id, master_vector, metadata)
        return True


    @staticmethod
    def validate_uniform(person_crop):
        """
        HÍBRIDO: 
        1. YOLO ubica las prendas con 'best.pt'.
        2. Recortamos cada prenda localizada y mapeamos a lógica interna (jacket, shirt, pants).
        3. Pasamos por ResNet para vector.
        4. Validamos el vector contra la DB.
        """
        results = yolo_clothing_structure_model(person_crop, device=DEVICE, verbose=False)
        
        detected_clothing_info = {} # Guardaremos {'jacket': recorte_bgr, 'pants': recorte_bgr, ...}

        # Paso 1: Localizar la ropa y adaptar las clases
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                internal_class = None
                if class_name in ['Coat', 'padded jacket']:
                    internal_class = 'jacket'
                elif class_name == 'clothes top':
                    internal_class = 'shirt'
                elif class_name == 'Pants':
                    internal_class = 'pants'
                
                # Solo guardar si es una prenda de interés para el uniforme (ignorar Skirt, Shoes, etc.)
                if internal_class:
                    # Guardamos el recorte exacto de esta prenda
                    detected_clothing_info[internal_class] = person_crop[y1:y2, x1:x2]

        has_top_valid = False
        has_bottom_valid = False
        details_log = []

        # Paso 2 y 3: Validar Prenda Superior (Jacket o Shirt)
        # 2a: Intentar con Jacket primero
        if 'jacket' in detected_clothing_info:
            jac_vector = ClothingEngine._get_embedding(detected_clothing_info['jacket'])
            match = search_uniform_by_vector(jac_vector)
            
            # Ajustar la tolerancia según pruebas (0.35 para colores/texturas ResNet suele ir bien)
            if match and match['distance'] < 0.35 and match['metadata'].get('tipo') == 'jacket':
                has_top_valid = True
                details_log.append("Chumpa OK")
            else:
                details_log.append("Chumpa No-Oficial")

        # 2b: Si no lleva chumpa válida, revisar si lleva la camisa oficial
        if not has_top_valid and 'shirt' in detected_clothing_info:
            shirt_vector = ClothingEngine._get_embedding(detected_clothing_info['shirt'])
            match = search_uniform_by_vector(shirt_vector)
            
            if match and match['distance'] < 0.35 and match['metadata'].get('tipo') == 'shirt':
                has_top_valid = True
                details_log.append("Camisa OK")
            else:
                details_log.append("Camisa No-Oficial")

        # Paso 4: Validar Prenda Inferior (Pants) obligatoriamente
        if 'pants' in detected_clothing_info:
            pants_vector = ClothingEngine._get_embedding(detected_clothing_info['pants'])
            match = search_uniform_by_vector(pants_vector)
            
            if match and match['distance'] < 0.35 and match['metadata'].get('tipo') == 'pants':
                has_bottom_valid = True
                details_log.append("Pantalón OK")
            else:
                details_log.append("Pantalón No-Oficial")
        else:
             details_log.append("Sin Pantalón Visible")

        is_full_uniform_valid = has_top_valid and has_bottom_valid
        
        if not details_log:
            details_log.append("Ropa no detectada por YOLO")

        return is_full_uniform_valid, " | ".join(details_log)