from db.connection import get_uniforms_collection


def search_uniform_by_vector(vector: list, tipo: str = None):
    """
    Busca el vector de uniforme más cercano en ChromaDB.

    Parámetro `tipo`:
      Si se especifica, ChromaDB filtra la colección a solo los documentos cuyo
      metadata["tipo"] coincida (ej. "jacket", "shirt", "pants").

      Por qué importa: sin filtro, una chumpa oficial podría quedar más cerca del
      vector promedio de camisa que del de chumpa → se rechaza incorrectamente.
      Con filtro por tipo, la comparación es siempre dentro de la misma categoría.
    """
    collection = get_uniforms_collection()
    query_kwargs = {"query_embeddings": [vector], "n_results": 1}
    if tipo:
        query_kwargs["where"] = {"tipo": tipo}
    try:
        results = collection.query(**query_kwargs)
    except Exception:
        # La colección puede estar vacía o el filtro no retorna resultados
        return None

    if not results["ids"] or not results["ids"][0]:
        return None

    return {
        "uniform_id": results["ids"][0][0],
        "metadata":   results["metadatas"][0][0],
        "distance":   results["distances"][0][0],
    }


def save_uniform_vector(uniform_id: str, vector: list, metadata: dict):
    """Inserta un vector. Falla si el ID ya existe (usar en contextos sin re-run)."""
    collection = get_uniforms_collection()
    collection.add(ids=[uniform_id], embeddings=[vector], metadatas=[metadata])
    return True


def upsert_uniform_vector(uniform_id: str, vector: list, metadata: dict):
    """Inserta o actualiza un vector. Idempotente para re-ejecuciones del seeder."""
    collection = get_uniforms_collection()
    collection.upsert(ids=[uniform_id], embeddings=[vector], metadatas=[metadata])
    return True


def search_uniform_by_vector_topk(vector: list, tipo: str = None, k: int = 5):
    """
    Top-k nearest neighbor search for consensus-based matching.
    Uses median distance over k results instead of single best match,
    making the scoring robust against individual outlier vectors.
    """
    collection = get_uniforms_collection()
    query_kwargs = {"query_embeddings": [vector], "n_results": k}
    if tipo:
        query_kwargs["where"] = {"tipo": tipo}
    try:
        results = collection.query(**query_kwargs)
    except Exception:
        return None

    if not results["ids"] or not results["ids"][0]:
        return None

    distances = results["distances"][0]
    sorted_d = sorted(distances)
    return {
        "top_distance": sorted_d[0],
        "mean_distance": sum(sorted_d) / len(sorted_d),
        "median_distance": sorted_d[len(sorted_d) // 2],
        "distances": sorted_d,
        "metadata": results["metadatas"][0][0],
    }
