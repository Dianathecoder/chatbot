import requests
from flask import Flask, request, jsonify
import faiss
from rag_utils import load_index, embedding_model
import numpy as np

app = Flask(__name__)

# 1. CARGAR DATOS
index, chunks = load_index()

LLM_API = "http://localhost:1234/v1/chat/completions"

# 2. DICCIONARIO Y FUNCIÓN (Esto es lo que te faltaba o estaba mal puesto)
ERRORS = {
    "spageti": "spaghetti", "paeya": "paella", "cocreta": "croqueta",
    "piza": "pizza", "recepi": "recipe"
}

def preprocess(text):
    text = text.lower().strip()
    for typo, fix in ERRORS.items():
        text = text.replace(typo, fix)
    return text

# 3. LÓGICA DE VALIDACIÓN
def es_respuesta_valida(respuesta, contexto_nombre):
    res_lc = respuesta.lower()
    negaciones = ["i don't know", "no tengo información", "sorry", "lo siento"]
    if any(neg in res_lc for neg in negaciones):
        return False
    if len(respuesta) < 20:
        return False
    if contexto_nombre.lower() not in res_lc:
        return False
    return True

def escalar_a_humano(pregunta):
    return "Chef's Table: I'm sorry, I don't have that specific recipe in my book. Would you like to talk to our head chef?"

# 4. RUTA DEL CHAT
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_msg = data.get("message", "")
        
        # Preprocesar
        clean_query = preprocess(user_msg)
        
        # RAG
        query_vec = embedding_model.encode([clean_query], convert_to_numpy=True).astype("float32")
        distances, indices = index.search(query_vec, 1)
        
        if indices[0][0] == -1 or distances[0][0] > 1.2:
            return jsonify({"response": escalar_a_humano(user_msg), "decision": "escalado_por_distancia"})

        recipe = chunks[indices[0][0]]
        
        # IMPORTANTE: Usamos 'titulo', 'ingredientes' e 'instrucciones' de tu recetas.json
        recipe_name = recipe.get('titulo', 'Recipe')
        recipe_ing = ", ".join(recipe.get('ingredientes', []))
        recipe_steps = ". ".join(recipe.get('instrucciones', []))

        prompt = (
            "You are a Michelin-star Chef. Answer in English using ONLY this context:\n"
            f"Recipe: {recipe_name}\nIngredients: {recipe_ing}\nSteps: {recipe_steps}\n\n"
            f"Question: {user_msg}\nAnswer:"
        )

        payload = {
            "model": "llama-3.2-3b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }

        r = requests.post(LLM_API, json=payload, timeout=10)
        response_text = r.json()["choices"][0]["message"]["content"]
        
        if es_respuesta_valida(response_text, recipe_name):
            return jsonify({"response": response_text, "status": "success"})
        else:
            return jsonify({"response": escalar_a_humano(user_msg), "status": "escalado_por_validacion"})
            
    except Exception as e:
        return jsonify({"response": "Technical difficulties.", "error": str(e)})

if __name__ == "__main__":
    app.run(port=5000, debug=True)