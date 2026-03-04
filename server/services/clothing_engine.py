import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from ultralytics import YOLO
import cv2
import numpy as np

from db.cruds.crud_uniform import save_uniform_vector, search_uniform_by_vector

# Carga en memoria global
DEVICE = '0' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO('yolov8n.pt')
resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
resnet_model.eval()

class ClothingEngine:

    @staticmethod
    def extract_torso(frame):
        """Detecta a la persona y recorta el torso"""
        results = yolo_model(frame, classes=[0], device=DEVICE, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                y_start = y1 + int((y2 - y1) * 0.20)
                return frame[y_start:y2, x1:x2]
        return None

    @staticmethod
    def get_clothing_embedding(cropped_image):
        """Convierte el recorte de ropa en un vector matemático"""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        img_tensor = transform(cropped_image).unsqueeze(0)
        with torch.no_grad():
            vector = resnet_model(img_tensor).flatten().tolist()
        return vector

    @staticmethod
    def register_uniform(uniform_id: str, uniform_type: str, images_list: list):
        """Registra un nuevo uniforme promediando múltiples imágenes (Vector Maestro)"""
        vectors_buffer = []

        for image_path in images_list:
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"[WARNING] Imagen no encontrada: {image_path}")
                continue

            torso = ClothingEngine.extract_torso(frame)
            if torso is not None:
                vector = ClothingEngine.get_clothing_embedding(torso)
                vectors_buffer.append(vector)
            else:
                print(f"[ERROR] YOLO no detectó persona en: {image_path}")

        if not vectors_buffer:
            raise ValueError("[ERROR CRITICO] No se pudo extraer ropa de ninguna imagen.")

        # Vector Maestro (Promedio matemático)
        master_vector = np.mean(vectors_buffer, axis=0).tolist()
        
        metadata = {"tipo": uniform_type, "valido": True}
        save_uniform_vector(uniform_id, master_vector, metadata)
        
        print(f"[SUCCESS] Uniforme '{uniform_type}' registrado exitosamente con ID '{uniform_id}'.")
        return True