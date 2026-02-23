import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_data")

def get_db_client():
    """
    Inicializa y devuelve el cliente de ChromaDB persistente.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client

def get_collection(collection_name: str):
    """
    Obtiene o crea una colección específica (ej. 'alumnos' o 'uniformes').
    """
    client = get_db_client()
    return client.get_or_create_collection(name=collection_name)