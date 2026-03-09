import chromadb
import os

from chromadb.config import Settings

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")
client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))

def get_faces_collection():
    return client.get_or_create_collection(name="student_faces", metadata={"hnsw:space": "l2"})

def get_uniforms_collection():
    return client.get_or_create_collection(name="uniform_catalog", metadata={"hnsw:space": "cosine"})