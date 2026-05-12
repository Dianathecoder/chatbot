import requests
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify
import faiss
from rag_utils import load_index, embedding_model
import numpy as np

app = Flask(__name__)
index, chunks = load_index()
LLM_API = "http://localhost:1234/v1/chat/completions"

logging.basicConfig(
    filename="escalados.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

sesiones = {}

# --- CAMBIO 1: ESTADOS ACTUALIZADOS CON LA PREGUNTA DE TIEMPO ---
ESTADOS = {
    "inicio":       "¿Qué tipo de cocina te apetece? Tenemos: Italiana o Española.",
    "tipo_elegido": "¿Tienes algún ingrediente principal en mente? (pasta, carne, tomate...)",
    "tiempo":       "¿Buscas algo rápido (menos de 30 min) o tienes tiempo para una receta elaborada?",
    "ingrediente":  "¿Tienes alguna restricción dietética? (vegetariano, low-carb... o escribe ninguna)",
}

# --- CAMBIO 2: CATEGORÍAS ADAPTADAS A TUS NUEVOS CHUNKS ---
CATEGORIAS = {
    "española": ["paella", "tortilla", "gazpacho"],
    "italiana": ["lasagna", "gnocchi", "bolognese", "sandwich", "pasta"],
}

def preprocesar_input(texto):
    texto = texto.lower().strip()
    if "espanola" in texto: texto = "española"
    return texto

def recuperar_receta(query):
    vec = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(vec, 1)
    idx = indices[0][0]
    dist = float(distances[0][0])
    if idx == -1 or dist > 1.5:
        return None, dist
    return chunks[idx], dist

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
        receta, dist = recuperar_receta(busqueda)

        if not receta:
            return {"response": "No encontré nada específico. ¿Deseas hablar con el Chef humano?"}

        # CAMBIO 4: MAPEO DE CLAVES DEL NUEVO JSON (titulo, instrucciones)
        nombre = receta.get("titulo", "Receta Especial")
        ings = ", ".join(receta.get("ingredientes", []))
        pasos = " ".join(receta.get("instrucciones", []))
        
        prompt = (
            f"Eres un Chef Michelin. Presenta esta receta: {nombre}.\n"
            f"Contexto: {ings}. Pasos: {pasos}.\n"
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