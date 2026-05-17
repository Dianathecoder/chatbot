import requests
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify
import faiss
from rag_utils import load_index, embedding_model, search
import numpy as np

app = Flask(__name__)
index, chunks = load_index()
LLM_API = "http://localhost:1234/v1/chat/completions"
DEFAULT_TOP_K = 3

logging.basicConfig(
    filename="escalados.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

sesiones = {}

# ESTADOS ACTUALIZADOS CON LA PREGUNTA DE TIEMPO ---
ESTADOS = {
    "inicio":       "¿Qué tipo de cocina te apetece? Tenemos: Italiana o Española.",
    "tipo_elegido": "¿Tienes algún ingrediente principal en mente? (pasta, carne, tomate...)",
    "tiempo":       "¿Buscas algo rápido (menos de 30 min) o tienes tiempo para una receta elaborada?",
    "ingrediente":  "¿Tienes alguna restricción dietética? (vegetariano, low-carb... o escribe ninguna)",
}

# CATEGORÍAS ADAPTADAS A TUS NUEVOS CHUNKS ---
CATEGORIAS = {
    "española": ["paella", "tortilla", "gazpacho"],
    "italiana": ["lasagna", "gnocchi", "bolognese", "sandwich", "pasta"],
}

def preprocesar_input(texto):
    texto = texto.lower().strip()
    if "espanola" in texto: texto = "española"
    return texto

def recuperar_receta(query, top_k=1):
    results = search(index, chunks, query, top_k=top_k)
    if not results:
        return [], []
    return results, [item["score"] for item in results]

def llamar_llm(prompt):
    payload = {
        "model": "llama-3.2-3b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 500,
    }
    try:
        r = requests.post(LLM_API, json=payload, timeout=20)
        return r.json()["choices"][0]["message"]["content"]
    except:
        return "Lo siento, el Chef no puede responder ahora. Revisa si LM Studio está encendido."

def manejar_pregunta(session_id, mensaje):
    limpio = preprocesar_input(mensaje)
    sesion = sesiones.setdefault(session_id, {"estado": "inicio", "datos": {}})

    # INICIO / REINICIO
    if "hola" in limpio or "reset" in limpio:
        sesiones[session_id] = {"estado": "esperando_tipo", "datos": {}}
        return {"response": "¡Bienvenido! " + ESTADOS["inicio"]}

    # 1. SELECCIÓN DE TIPO
    if sesion["estado"] == "esperando_tipo":
        if "italiana" in limpio or "española" in limpio:
            sesion["datos"]["tipo"] = "italiana" if "italiana" in limpio else "española"
            sesion["estado"] = "esperando_ingrediente"
            return {"response": ESTADOS["tipo_elegido"]}
        return {"response": "Por favor, elige entre Italiana o Española."}

    # 2. SELECCIÓN DE INGREDIENTE
    if sesion["estado"] == "esperando_ingrediente":
        sesion["datos"]["ingrediente"] = limpio
        sesion["estado"] = "esperando_tiempo" # CAMBIO 3: PASA A TIEMPO
        return {"response": ESTADOS["tiempo"]}

    # 3. SELECCIÓN DE TIEMPO (NUEVA PREGUNTA)
    if sesion["estado"] == "esperando_tiempo":
        sesion["datos"]["tiempo"] = limpio
        sesion["estado"] = "esperando_dieta"
        return {"response": ESTADOS["ingrediente"]}

    # 4. RESTRICCIÓN Y BÚSQUEDA FINAL
    if sesion["estado"] == "esperando_dieta":
        sesion["datos"]["dieta"] = limpio
        
        # Búsqueda RAG combinando los datos recogidos
        busqueda = f"{sesion['datos']['tipo']} {sesion['datos']['ingrediente']}"
        recetas, distances = recuperar_receta(busqueda, top_k=DEFAULT_TOP_K)

        if not recetas:
            return {"response": "No encontré nada específico. ¿Deseas hablar con el Chef humano?"}

        context = []
        for index, receta in enumerate(recetas, start=1):
            nombre = receta.get("titulo", "Receta Especial")
            ingredientes = ", ".join(receta.get("ingredientes", []))
            pasos = " ".join(receta.get("instrucciones", []))
            score = receta.get("score", 0.0)
            context.append(
                f"Receta {index}: {nombre}\nIngredientes: {ingredientes}\nPasos: {pasos}\nScore FAISS: {score:.4f}"
            )

        prompt = (
            "Eres un Chef Michelin. Usa el siguiente contexto para responder con elegancia y sabor.\n\n"
            f"{chr(10).join(context)}\n\n"
            f"El cliente tiene {sesion['datos']['tiempo']} y sigue una dieta {sesion['datos']['dieta']}.\n"
            "Responde en español de forma elegante."
        )
        
        respuesta = llamar_llm(prompt)
        sesion["estado"] = "finalizado"
        return {"response": respuesta}

    return {"response": "Dime 'hola' para empezar de nuevo."}

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    return jsonify(manejar_pregunta(data.get("session_id", "123"), data.get("message", "")))

if __name__ == "__main__":
    app.run(port=5000, debug=True)