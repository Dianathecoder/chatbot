import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = "index_recipes"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(MODEL_NAME)

def load_recipes(json_file="recetas.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

def save_index(index, chunks):
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

def load_index():
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError("No existe el índice. Ejecuta build_index.py")
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks