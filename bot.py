import requests
from flask import Flask, request, jsonify
import faiss
from rag_utils import load_index, embedding_model
import numpy as np

app = Flask(__name__)
index, chunks = load_index()

LLM_API = "http://localhost:1234/v1/chat/completions"

# DICCIONARIO DE ERRORES (Preprocesado)
ERRORS = {
    "spageti": "spaghetti", "spaguetti": "spaghetti", "spagety": "spaghetti",
    "tagliateli": "tagliatelle", "taliatele": "tagliatelle",
    "fuchili": "fusilli", "fucilli": "fusilli",
    "rizoto": "risotto", "risoto": "risotto",
    "gnonchi": "gnocchi", "noqui": "gnocchi",
    "lasaña": "lasagna", "lasagne": "lasagna",
    "piza": "pizza", "pisa": "pizza",
    "focacha": "focaccia", "focacia": "focaccia",
    "paeya": "paella", "paeia": "paella",
    "tortia": "tortilla", "tortiya": "tortilla",
    "crocreta": "croqueta", "cocreta": "croqueta",
    "recepi": "recipe", "recipi": "recipe",
    "ingredents": "ingredients", "ingridients": "ingredients"
}

def preprocess(text):
    text = text.lower().strip()
    for typo, fix in ERRORS.items():
        text = text.replace(typo, fix)
    return text

def es_respuesta_valida(respuesta, contexto_nombre):
    """
    LÓGICA DE DECISIÓN POST-LLM:
    Determina si la respuesta generada por la IA es aceptable.
    """
    res_lc = respuesta.lower()
    
    # 1. Filtro de Negación: Si el LLM admite que no sabe
    negaciones = ["i don't know", "no tengo información", "sorry", "lo siento"]
    if any(neg in res_lc for neg in negaciones):
        return False

    # 2. Filtro de Longitud: Si la respuesta es demasiado corta (posible error)
    if len(respuesta) < 20:
        return False

    # 3. Filtro de Alucinación: ¿Menciona al menos el nombre del plato?
    if contexto_nombre.lower() not in res_lc:
        return False

    return True

def escalar_a_humano(pregunta):
    """
    LÓGICA DE ESCALADO:
    Mensaje estándar cuando la IA no puede o no debe responder.
    """
    return "Chef's Table: I'm sorry, I don't have that specific recipe in my book. Would you like to talk to our head chef?"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_msg = data.get("message", "")
    
    # 1. Preprocesar
    clean_query = preprocess(user_msg)
    
    # 2. RAG Semántico
    query_vec = embedding_model.encode([clean_query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, 1)
    
    # LÓGICA DE DECISIÓN PRE-LLM (Umbral de distancia)
    # Si la distancia es > 1.2, el tema está muy alejado de lo que conocemos
    if indices[0][0] == -1 or distances[0][0] > 1.2:
        return jsonify({
            "response": escalar_a_humano(user_msg),
            "decision": "escalado_por_distancia"
        })

    recipe = chunks[indices[0][0]]
    
    # 3. Prompting
    prompt = (
        "You are a Michelin-star Chef. Answer in English using ONLY this context:\n"
        f"Recipe: {recipe['name']}\nIngredients: {recipe.get('ingredients')}\nSteps: {recipe.get('instructions')}\n\n"
        f"Question: {user_msg}\nAnswer:"
    )

    # 4. Llamada a LM Studio
    payload = {
        "model": "llama-3.2-3b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }

    try:
        r = requests.post(LLM_API, json=payload, timeout=10)
        response_text = r.json()["choices"][0]["message"]["content"]
        
        # 5. LÓGICA DE DECISIÓN FINAL (Validación de respuesta)
        if es_respuesta_valida(response_text, recipe['name']):
            return jsonify({
                "response": response_text,
                "status": "success"
            })
        else:
            return jsonify({
                "response": escalar_a_humano(user_msg),
                "status": "escalado_por_validacion"
            })
            
    except Exception as e:
        return jsonify({"response": "Technical difficulties in the kitchen.", "error": str(e)})

if __name__ == "__main__":
    app.run(port=5000, debug=True)