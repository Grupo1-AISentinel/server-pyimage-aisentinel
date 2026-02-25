import face_recognition
import sys
import os

# Configurar el path para importar módulos del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.biometric_engine import BiometricEngine

def test_solo_reconocimiento():
    print("\n--- INICIANDO TEST DE RECONOCIMIENTO ESPECÍFICO ---")

    # DEFINIR LOS DATOS QUE ESPERAMOS ENCONTRAR
    target_carnet = None
    target_name = "Desconocido"
    image_rel_path = "img/crh.jpeg"

    # UBICAR LA IMAGEN DINÁMICAMENTE
    base_dir = os.path.dirname(__file__)
    image_path = os.path.join(base_dir, image_rel_path)

    if not os.path.exists(image_path):
        print(f"ERROR CRÍTICO: No se encuentra la imagen en: {image_path}")
        print("Asegurarse de que el archivo exista dentro de la carpeta 'tests/img/'")
        return

    try:
        # 3. CARGAR LA IMAGEN
        print("🔄 Cargando imagen...")
        fake_frame = face_recognition.load_image_file(image_path)

        # EL MOTOR BIOMÉTRICO
        resultados = BiometricEngine.recognize_faces_in_frame(fake_frame)

        # Validar Resultados
        if not resultados:
            print("FALLO: El motor no detectó ningún rostro en la imagen.")
            return

        detected_correctly = False
        print("\n📊 RESULTADOS DEL ANÁLISIS:")
        for i, res in enumerate(resultados):
            print(f"   Rostro #{i+1}:")
            print(f"     - Identidad Detectada: {res['identity']}")
            print(f"     - Carnet ID: {res['student_id']}")
            print(f"     - Confianza: {res['confidence']}")

            if res['student_id'] == target_carnet:
                detected_correctly = True

        print("\n-----------------------------------")
        if detected_correctly:
            print(f"✅ PRUEBA EXITOSA: El sistema reconoció correctamente a {target_name}.")
            print(f"   --> Distancia Real (ChromaDB): {match['distance']}")
        else:
            print(f"PRUEBA FALLIDA: Se detectaron rostros, pero ninguno es {target_name}.")
        print("-----------------------------------\n")

    except Exception as e:
        print(f"ERROR RUNTIME DURANTE EL TEST: {e}")

if __name__ == "__main__":
    # Se requiere que el usuario ya esté en la DB para que esto funcione.
    test_solo_reconocimiento()