from db.connection import get_uniforms_collection

def search_uniform_by_vector(vector: list):
    collection = get_uniforms_collection()
    results = collection.query(query_embeddings=[vector], n_results=1)
    if not results['ids'] or not results['ids'][0]: 
        return None
    return {
        "uniform_id": results['ids'][0][0],
        "metadata": results['metadatas'][0][0],
        "distance": results['distances'][0][0]
    }
    
def save_uniform_vector(uniform_id: str, vector: list, metadata: dict):
    collection = get_uniforms_collection()
    collection.add(ids=[uniform_id], embeddings=[vector], metadatas=[metadata])
    return True