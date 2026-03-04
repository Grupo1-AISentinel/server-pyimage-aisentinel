import face_recognition
import numpy as np
import os
from db.cruds.crud_student import save_student_vector, search_student_by_vector


# Motor de reconocimiento facial: registra y reconoce rostros usando ChromaDB
class BiometricEngine:

    @staticmethod
    def register_face(student_id: str, full_name: str, images_list: list):
        
        vectors_buffer = []         

        model = os.getenv("MODEL_FACE_RECOGNITION", "cnn")
        
        for image_file in images_list:
           try:
                image = face_recognition.load_image_file(image_file)
                face_locations = face_recognition.face_locations(image, model=model)
                
                # Solo aceptar si sale un rostro claro en la imagen
                if len(face_locations) == 1:
                    enconding = face_recognition.face_encodings(image, face_locations, num_jitters=10)[0]
                    vectors_buffer.append(enconding.tolist())
                else:
                    print(f"Advertencia: La imagen '{image_file}' no tiene un rostro claro. Se omitirá.")
           except Exception as e:
                print(f"Error al procesar la imagen '{image_file}': {e}. Se omitirá.")
                
        if not vectors_buffer:
            raise ValueError("No se pudo registrar el rostro. Asegúrese de proporcionar al menos una imagen clara con un rostro visible.")
            
        if len(vectors_buffer) < 3:
                print("Advertencia: Se recomienda al menos 3 imágenes claras para un mejor reconocimiento. Se procederá con las imágenes disponibles.")
            
        master_vector = np.mean(vectors_buffer, axis=0).tolist()
        
        metadata = {
            "student_id": student_id,
            "full_name": full_name
        }
        
        save_student_vector(student_id, master_vector, metadata)
        print(f"Rostro registrado exitosamente para el estudiante '{full_name}' con ID '{student_id}'.")
    
        return True

    # Reconocer rostros en un frame: recibe un frame RGB y devuelve lista de resultados
    @staticmethod
    def recognize_faces_in_frame(frame_rgb):
        model = os.getenv("MODEL_FACE_RECOGNITION", "cnn")
        face_locations = face_recognition.face_locations(frame_rgb, model=model)
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        results = []

        for (top, right, bottom, left), face_vector in zip(
            face_locations, face_encodings
        ):
            match = search_student_by_vector(face_vector.tolist())

            identity = "Desconocido"
            student_id = None
            color = (0, 0, 255)
            confidence = 0.0

            threshold = float(os.getenv("FACE_RECOGNITION_THRESHOLD", 0.45))

            if match and match["distance"] < threshold:
                identity = match["metadata"].get("full_name", "Estudiante")

                student_id = match["metadata"].get(
                    "student_id", match.get("student_id")
                )
                color = (0, 255, 0)
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
