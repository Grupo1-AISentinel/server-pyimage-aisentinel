import face_recognition
import os
from db.connection import save_student_vector, search_student_by_vector


# Motor de reconocimiento facial: registra y reconoce rostros usando ChromaDB
class BiometricEngine:

    @staticmethod
    def register_face(student_id: str, full_name: str, image_file):
        try:
            image = face_recognition.load_image_file(image_file)
        except Exception:
            raise ValueError("El archivo no es una imagen válida.")

        model = os.getenv("MODEL_FACE_RECOGNITION", "cnn")
        face_locations = face_recognition.face_locations(image, model=model)

        if len(face_locations) != 1:
            raise ValueError(
                f"Se detectaron {len(face_locations)} rostros. Usa una foto con UN solo rostro."
            )

        face_encodings = face_recognition.face_encodings(
            image, face_locations, num_jitters=10
        )

        face_vector = face_encodings[0].tolist()

        metadata = {"full_name": full_name, "type": "FACE_VECTOR", "active": True}

        save_student_vector(student_id, face_vector, metadata)
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
