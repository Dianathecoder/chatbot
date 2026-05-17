from rag_utils import load_recipes, save_index, embedding_model
import faiss
import numpy as np

def build():
    print("Leyendo recetas.json...")
    recipes = load_recipes()
    
    texts = [f"{r.get('titulo', 'Sin título')} {' '.join(r.get('ingredientes', []))}" for r in recipes]
    
    print("Creando vectores (embeddings)...")
    embeddings = embedding_model.encode(texts, convert_to_numpy=True).astype("float32")
    
    print("Construyendo índice FAISS...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    save_index(index, recipes)
    print("¡Índice creado con éxito!")

if __name__ == "__main__":
    build()