from datetime import datetime
from db.connection import get_faces_collection

def save_student_vector(student_id: str, vector: list, metadata: dict):
    collection = get_faces_collection()
    metadata["created_at"] = datetime.now().isoformat()
    metadata["student_id"] = student_id 
    collection.add(ids=[student_id], embeddings=[vector], metadatas=[metadata])
    return True

def search_student_by_vector(vector: list, limit: int = 1):
    collection = get_faces_collection()
    results = collection.query(query_embeddings=[vector], n_results=limit)
    if not results['ids'] or not results['ids'][0]: 
        return None
    return {
        "student_id": results['ids'][0][0],
        "metadata": results['metadatas'][0][0],
        "distance": results['distances'][0][0]
    }