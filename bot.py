import requests
from flask import Flask, request, jsonify, stream_with_context, Response
from rapidfuzz import process
import faiss
from rag_utils import load_index, embedding_model
import numpy as np
from rapidfuzz import process




app = Flask(__name__)
index, chunks = load_index()

LLM_API = "http://localhost:1234/v1/chat/completions"

# DICCIONARIO DE ERRORES (Preprocesado)
ERRORS = {
    # Pasta & Italian
    "spageti": "spaghetti",
    "spaguetti": "spaghetti",
    "spagety": "spaghetti",
    "tagliateli": "tagliatelle",
    "taliatele": "tagliatelle",
    "fuchili": "fusilli",
    "fucilli": "fusilli",
    "penne": "penne", # often misspelled as pene (careful there!)
    "rizoto": "risotto",
    "risoto": "risotto",
    "gnonchi": "gnocchi",
    "noqui": "gnocchi",
    "lasaña": "lasagna",
    "lasagne": "lasagna",
    "piza": "pizza",
    "pisa": "pizza",
    "focacha": "focaccia",
    "focacia": "focaccia",
    
    # Spanish Dishes
    "paeya": "paella",
    "paeia": "paella",
    "tortia": "tortilla",
    "tortiya": "tortilla",
    "gazpacho": "gazpacho",
    "gaspacho": "gazpacho",
    "choriso": "chorizo",
    "choriço": "chorizo",
    "crocreta": "croqueta",
    "cocreta": "croqueta",
    
    # General Ingredients & Cooking Terms
    "recepi": "recipe",
    "recipi": "recipe",
    "ingredents": "ingredients",
    "ingridients": "ingredients",
    "vegies": "vegetables",
    "vegtables": "vegetables",
    "chicken": "chicken", # chicen, chiken
    "chiken": "chicken",
    "potatos": "potatoes",
    "tomatos": "tomatoes",
    "oilve oil": "olive oil",
    "vengar": "vinegar",
    "garlic": "garlic", # garlik
    "garlik": "garlic",
    
    # Verbs/Actions
    "how to": "how to",
    "makeing": "making",
    "cookin": "cooking",
    "prepair": "prepare",
    "bakeing": "bakery"
}

def preprocess(text):
    text = text.lower().strip()
    for typo, fix in ERRORS_DICT.items():
        text = text.replace(typo, fix)
    return text

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_msg = data.get("message", "")
    
    # 1. Preprocesar
    clean_query = preprocess(user_msg)
    
    # 2. RAG Semántico
    query_vec = embedding_model.encode([clean_query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, 1)
    
    if indices[0][0] == -1 or distances[0][0] > 1.5:
        return jsonify({"response": "I'm sorry, I don't have that recipe."})

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
        r = requests.post(LLM_API, json=payload)
        response_text = r.json()["choices"][0]["message"]["content"]
        return jsonify({"response": response_text})
    except:
        return jsonify({"response": "Error: Is LM Studio running?"})

if __name__ == "__main__":
    app.run(port=5000)