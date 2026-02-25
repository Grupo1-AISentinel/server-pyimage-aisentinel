import chromadb
import os
from datetime import datetime

# Configuración de persistencia
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Obtener siempre la colección
def get_faces_collection():
    return client.get_or_create_collection(
        name="student_faces",
        metadata={"hnsw:space": "l2"}
    )
    
# Guardar el vector facial con su metadata asociada
def save_student_vector(student_id: str, vector: list, metadata: dict):
    
    collection = get_faces_collection()
    
    metadata["created_at"] = datetime.now().isoformat()
    metadata["student_id"] = student_id 
    
    collection.add(
        ids=[student_id],
        embeddings=[vector],
        metadatas=[metadata]
    )
    return True

# Buscar el vector más similar en la colección y devolver su metadata
def search_student_by_vector(vector: list, limit: int = 1):

    collection = get_faces_collection()
    
    results = collection.query(
        query_embeddings=[vector],
        n_results=limit
    )
    
    if not results['ids'] or not results['ids'][0]:
        return None

    return {
        "student_id": results['ids'][0][0],
        "metadata": results['metadatas'][0][0],
        "distance": results['distances'][0][0]
    }