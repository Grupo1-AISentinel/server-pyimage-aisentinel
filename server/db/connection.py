import chromadb
import os

from chromadb.config import Settings

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")
FACES_COLLECTION_NAME = os.getenv("FACES_COLLECTION_NAME", "student_faces")
UNIFORMS_COLLECTION_NAME = os.getenv("UNIFORMS_COLLECTION_NAME", "uniform_catalog")

print("[DEBUG] Inicializando ChromaDB PersistentClient...")
client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))

def get_faces_collection():
    return client.get_or_create_collection(
        name=FACES_COLLECTION_NAME,
        metadata={"hnsw:space": "l2"},
    )

def get_uniforms_collection():
    return client.get_or_create_collection(
        name=UNIFORMS_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )