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

def load_recipes(json_file="data/recetas.json"):
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

def search(index, chunks, query, top_k=1):
    if top_k <= 0:
        return []

    query_vector = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    search_k = min(top_k, len(chunks))
    distances, indices = index.search(query_vector, search_k)

    results = []
    for rank, chunk_index in enumerate(indices[0]):
        if chunk_index < 0 or chunk_index >= len(chunks):
            continue
        chunk = dict(chunks[chunk_index])
        chunk["score"] = float(distances[0][rank])
        results.append(chunk)

    return results