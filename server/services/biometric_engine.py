import face_recognition
import numpy as np
import os
import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from db.cruds.crud_student import save_student_vector, search_student_by_vector


# Carga en memoria global
DEVICE = '0' if torch.cuda.is_available() else 'cpu'

# Descargar modelo state-of-the-art de rostros (Altamente preciso y resistente a inclinaciones)
print("[INFO] Cargando/Descargando el modelo YOLO11 Face desde HuggingFace (AdamCodd)...")
try:
    hf_model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
    yolo_face_model = YOLO(hf_model_path)
except Exception as e:
    print(f"[ERROR CRÍTICO] No se pudo descargar el modelo de rostro de AdamCodd. Detalle: {e}")
    print("[INFO] Cayendo de vuelta al modelo YOLO11 genérico de rostros...")
    yolo_face_model = YOLO('yolo11n-face.pt')
# Motor de reconocimiento facial: registra y reconoce rostros usando ChromaDB
class BiometricEngine:
    
    @staticmethod
    def _get_face_locations_yolo(image_rgb):
        """Usa YOLO para encontrar exacto dónde están los rostros y devuelve el array (top, right, bottom, left)"""
        # imgsz=320: el frame ya llega a 320x240; evita que YOLO haga upscale interno a 640
        results = yolo_face_model(image_rgb, device=DEVICE, verbose=False, imgsz=320)
        face_locations = []
        for result in results:
            for box in result.boxes:
                # YOLO devuelve box: x1, y1, x2, y2
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # face_recognition espera: (top, right, bottom, left) que es (y1, x2, y2, x1)
                face_locations.append((y1, x2, y2, x1))
        return face_locations

    @staticmethod
    def register_face(student_id: str, full_name: str, images_list: list):
        
        vectors_buffer = []         
        
        for image_file in images_list:
           try:
                image = face_recognition.load_image_file(image_file)
                # Obtenemos locaciones con YOLO en vez de cnn o hog
                face_locations = BiometricEngine._get_face_locations_yolo(image)
                
                # Solo aceptar si sale un rostro claro y UNICO en la imagen
                if len(face_locations) == 1:
                    enconding = face_recognition.face_encodings(image, face_locations, num_jitters=10)[0]
                    vectors_buffer.append(enconding.tolist())
                elif len(face_locations) > 1:
                    print(f"Advertencia: La imagen '{image_file}' tiene múltiples rostros. Se omitirá para evitar mezcla de identidades.")
                else:
                    print(f"Advertencia: YOLO no detectó un rostro claro en '{image_file}'. Se omitirá.")
           except Exception as e:
                print(f"Error al procesar la imagen '{image_file}': {e}. Se omitirá.")
                
        if not vectors_buffer:
            raise ValueError("No se pudo registrar el rostro. Asegúrese de proporcionar al menos una imagen clara con un rostro visible.")
            
        if len(vectors_buffer) < 3:
            print("Advertencia: Se recomienda al menos 3 imágenes claras para un mejor perfilamiento. Se procederá con las imágenes disponibles.")
            
        master_vector = np.mean(vectors_buffer, axis=0).tolist()
        
        metadata = {
            "student_id": student_id,
            "full_name": full_name
        }
        
        save_student_vector(student_id, master_vector, metadata)
        print(f"Rostro registrado exitosamente para el estudiante '{full_name}' con ID '{student_id}'.")
    
        return True
    
    def register_face_from_socket(self, student_id: str, full_name: str, images_list: list):
        """
        Método especializado para registros vía Socket.io.
        """
        vectors_buffer = []
        
        print(f"[SOCKET-IA] Iniciando procesamiento biométrico para {full_name}...")

        for i, image_np in enumerate(images_list):
            try:
                # Usamos self para llamar al método de localización si no es estático
                # O BiometricEngine si el otro sigue siendo staticmethod
                face_locations = BiometricEngine._get_face_locations_yolo(image_np)
                
                if len(face_locations) == 1:
                    encoding = face_recognition.face_encodings(image_np, face_locations, num_jitters=1)[0]
                    vectors_buffer.append(encoding.tolist())
                    print(f"   ✅ Imagen {i+1}: Rostro detectado.")
                else:
                    print(f"   ⚠️ Imagen {i+1}: Se detectaron {len(face_locations)} rostros.")

            except Exception as e:
                print(f"   ❌ Error procesando imagen {i+1}: {e}")

        if not vectors_buffer:
            return False

        master_vector = np.mean(vectors_buffer, axis=0).tolist()
        
        metadata = {
            "student_id": student_id,
            "full_name": full_name,
            "source": "socket_bidirectional"
        }

        try:
            save_student_vector(student_id, master_vector, metadata)
            return True
        except Exception as e:
            print(f"❌ Error al guardar: {e}")
            return False

    # Reconocer rostros en un frame: recibe un frame RGB y devuelve lista de resultados
    @staticmethod
    def recognize_faces_in_frame(frame_rgb):
        
        # 1. Obtener cuadros usando YOLO (CERO falsos positivos)
        face_locations = BiometricEngine._get_face_locations_yolo(frame_rgb)
        
        if not face_locations:
            return [] # Retorno rápido si no hay rostros detectados por YOLO

        # 2. Extraer vectores exactamente de esos encuadres
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        results = []

        for (top, right, bottom, left), face_vector in zip(
            face_locations, face_encodings
        ):
            # 3. Buscar en la Base de Datos (Chroma)
            match = search_student_by_vector(face_vector.tolist())

            identity = "Desconocido"
            student_id = None
            color = (0, 0, 255) # Rojo si no reconoce
            confidence = 0.0

            threshold = float(0.30) # Reducido de .40 a .30 para evitar confusión entre hermanos o parientes

            if match and match["distance"] < threshold:
                identity = match["metadata"].get("full_name", "Estudiante")

                student_id = match["metadata"].get(
                    "student_id", match.get("student_id")
                )
                color = (0, 255, 0) # Verde si es válido
                confidence = round((1 - match["distance"]) * 100, 1)

            results.append(
                {
                    "location": (top, right, bottom, left),
                    "identity": identity,
                    "student_id": student_id,
                    "color": color,
                    "confidence": f"{confidence}%",
                }
            )

        return results
