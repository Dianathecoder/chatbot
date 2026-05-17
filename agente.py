from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any

from flask import Flask, jsonify, render_template_string, request

from rag_utils import load_index, embedding_model, search

from app.config import Config
from app.llm.llm_service import LLMConfig, LLMService

logging.basicConfig(
    filename="escalados.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

HTML_TEMPLATE = r"""
<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Chef Michelin RAG</title>
    <style>
      :root {
        --bg: #f5f7f8;
        --surface: #ffffff;
        --ink: #152025;
        --muted: #61717a;
        --line: #ccd6dc;
        --accent: #d35400;
        --accent-dark: #a04000;
        --soft: #fcf3ee;
        --code-bg: #101719;
        --code-ink: #eef7f4;
        --user-msg: #e8f0fe;
        --bot-msg: #fcf3ee;
      }

      * { box-sizing: border-box; }

      body {
        margin: 0;
        color: var(--ink);
        background: var(--bg);
        font-family: Arial, Helvetica, sans-serif;
      }

      button, input { font: inherit; }

      main {
        width: min(800px, 100%);
        min-height: 100vh;
        margin: 0 auto;
        padding: 18px;
      }

      h1 { margin-top: 0; margin-bottom: 6px; font-size: 28px; }
      p { margin-top: 0; line-height: 1.45; }

      .panel {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--surface);
        padding: 16px;
        margin-bottom: 16px;
      }

      .intro p { color: var(--muted); }

      .chat-box {
        display: flex;
        flex-direction: column;
        gap: 12px;
        max-height: 450px;
        overflow-y: auto;
        padding: 12px;
        border: 1px solid var(--line);
        border-radius: 8px;
        margin-bottom: 14px;
        background: #fafafa;
      }

      .msg {
        padding: 10px 14px;
        border-radius: 8px;
        max-width: 85%;
        line-height: 1.4;
      }

      .msg.user {
        align-self: flex-end;
        background: var(--user-msg);
        border-bottom-right-radius: 0;
      }

      .msg.bot {
        align-self: flex-start;
        background: var(--bot-msg);
        border-bottom-left-radius: 0;
        border: 1px solid #f6e3d7;
        white-space: pre-wrap; /* Respeta los saltos de línea de la receta */
      }

      .input-row {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 10px;
        margin-bottom: 14px;
      }

      input {
        width: 100%;
        min-height: 42px;
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 8px 10px;
      }

      button {
        min-height: 42px;
        border: 1px solid var(--accent);
        border-radius: 6px;
        padding: 8px 12px;
        color: #ffffff;
        background: var(--accent);
        cursor: pointer;
        font-weight: bold;
      }

      button:hover { background: var(--accent-dark); }
      
      button.secondary {
        color: var(--accent);
        background: #ffffff;
        font-weight: normal;
      }
      button.secondary:hover { background: var(--soft); }

      .examples {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
        gap: 8px;
        margin-bottom: 14px;
      }

      pre {
        overflow: auto;
        max-height: 250px;
        margin: 14px 0 0;
        border-radius: 6px;
        padding: 12px;
        color: var(--code-ink);
        background: var(--code-bg);
      }
    </style>
  </head>
  <body>
    <main>
      <section class="panel intro">
        <h1>Agente Chef Michelin</h1>
        <p>Tu asistente culinario RAG. Te hará unas preguntas para buscar la receta perfecta en nuestra base de datos.</p>
      </section>

      <section class="panel">
        <div class="chat-box" id="chat-box">
          <div class="msg bot">¡Hola! Escribe "hola" para iniciar la búsqueda de recetas.</div>
        </div>

        <div class="input-row">
          <input id="message-input" placeholder="Escribe tu respuesta..." aria-label="Mensaje" autocomplete="off">
          <button type="button" id="send-button">Enviar</button>
        </div>

        <div class="examples">
          <button type="button" class="secondary" data-message="hola">👋 Hola (Reset)</button>
          <button type="button" class="secondary" data-message="española">🥘 Española</button>
          <button type="button" class="secondary" data-message="italiana">🍝 Italiana</button>
          <button type="button" class="secondary" data-message="pasta">🍅 Pasta / Carne</button>
          <button type="button" class="secondary" data-message="menos de 30 min">⏱️ Rápido</button>
          <button type="button" class="secondary" data-message="ninguna">✅ Sin dieta</button>
        </div>

        <details>
          <summary style="cursor:pointer; color: var(--muted); font-weight: bold; margin-top: 10px;">Ver estado interno (Debug JSON)</summary>
          <pre><code id="result-json">{}</code></pre>
        </details>
      </section>
    </main>

    <script>
      const input = document.querySelector("#message-input");
      const sendButton = document.querySelector("#send-button");
      const chatBox = document.querySelector("#chat-box");
      const resultJson = document.querySelector("#result-json");
      
      const sessionId = Math.random().toString(36).substring(2, 15);

      function addMessage(text, sender) {
        const div = document.createElement("div");
        div.className = `msg ${sender}`;
        div.textContent = text;
        chatBox.appendChild(div);
        chatBox.scrollTop = chatBox.scrollHeight;
      }

      async function send(message = input.value) {
        if (!message.trim()) return;
        
        addMessage(message, "user");
        input.value = "";
        
        // Indicador de "Escribiendo..."
        const typingDiv = document.createElement("div");
        typingDiv.className = "msg bot typing";
        typingDiv.textContent = "El chef está cocinando tu respuesta...";
        chatBox.appendChild(typingDiv);
        chatBox.scrollTop = chatBox.scrollHeight;

        try {
          const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId, message })
          });
          
          const result = await response.json();
          chatBox.removeChild(typingDiv); // Quitar indicador
          addMessage(result.reply, "bot");
          resultJson.textContent = JSON.stringify(result, null, 2);
        } catch (err) {
          chatBox.removeChild(typingDiv);
          addMessage("Hubo un error de conexión con el servidor.", "bot");
        }
      }

      sendButton.addEventListener("click", () => send());
      input.addEventListener("keypress", (e) => { if (e.key === "Enter") send(); });
      
      document.querySelectorAll("[data-message]").forEach((button) => {
        button.addEventListener("click", () => { send(button.dataset.message); });
      });
    </script>
  </body>
</html>
"""

DEFAULT_TOP_K = 3

ERRORS = {
    "arros": "arroz",
    "poyo": "pollo",
    "spagueti": "espagueti",
    "espagheti": "espagueti",
    "zanoria": "zanahoria",
    "kueso": "queso",
    "huevoz": "huevos",
    "vejetariano": "vegetariano",
    "saludavle": "saludable",
    "rapido": "rápido",
    "cantid": "cantidad",
    "azucar": "azúcar"
}


def preprocesar_input(texto: str) -> str:
    texto = texto.lower().strip()
    texto = " ".join(texto.split())

    for mal, bien in ERRORS.items():
        texto = texto.replace(mal, bien)

    return texto


def build_prompt(contexto: str, pregunta: str) -> str:
    return (
        "Eres un agente de soporte técnico de atención al cliente. "
        "Responde usando SOLO la información del contexto. "
        "Si no puedes responder con seguridad, di que escalarás el caso a un humano.\n\n"
        f"Contexto:\n{contexto}\n\n"
        f"Pregunta: {pregunta}\n"
        "Respuesta:" 
    )


def es_respuesta_valida(respuesta: str, contexto: str) -> bool:
    if not contexto or len(contexto.strip()) == 0:
        return False
    if len(respuesta.strip()) < 20:
        return False
    low = respuesta.lower()
    if "no tengo información" in low or "no sé" in low or "no puedo" in low:
        return False
    return True


def escalar_a_humano(pregunta: str) -> str:
    return (
        "No he podido resolver tu problema con la información disponible. "
        "Te paso con un agente humano para que te ayude mejor."
    )

ESTADOS = {
    "inicio":       "¿Qué tipo de cocina te apetece? Tenemos: Italiana o Española.",
    "tipo_elegido": "¿Tienes algún ingrediente principal en mente? (pasta, sausage, beef...)",
    "tiempo":       "Perfecto. ¿De cuánto tiempo dispones para cocinar? (Ej: rápido 30 min, 1 hora, sin prisa...)",
    "ingrediente":  "¿Tienes alguna restricción dietética? (Alta en Proteína, Low-Carb... o escribe ninguna)",
}

class ChefRagAgent:
    def __init__(self, llm_service: LLMService | None):
        self.llm_service = llm_service
        self.index, self.chunks = load_index()
        self.sesiones = {}

    def preprocesar_input(self, texto: str) -> str:
        texto = texto.lower().strip()
        if "espanola" in texto: 
            texto = "española"
        return texto

    def recuperar_chunks(self, query: str, top_k: int = 1) -> tuple[list[dict], list[float]]:
        if top_k <= 0:
            return [], []

        results = search(self.index, self.chunks, query, top_k=top_k)
        if not results:
            return [], []

        return results, [chunk["score"] for chunk in results]

    def build_context(self, recetas: list[dict]) -> str:
        context_rows = []
        for index, receta in enumerate(recetas, start=1):
            nombre = receta.get("titulo", "Receta")
            ingredientes = ", ".join(receta.get("ingredientes", []))
            pasos = " ".join(receta.get("instrucciones", []))
            score = receta.get("score", 0.0)
            context_rows.append(
                f"Receta {index}: {nombre}\nIngredientes: {ingredientes}\nPasos: {pasos}\nScore FAISS: {score:.4f}"
            )
        return "\n\n".join(context_rows)

    def run(self, session_id: str, mensaje: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        limpio = preprocesar_input(mensaje)
        sesion = self.sesiones.setdefault(session_id, {"estado": "inicio", "datos": {}})

        # Reset forzado
        if "hola" in limpio or "reset" in limpio:
            self.sesiones[session_id] = {"estado": "esperando_tipo", "datos": {}}
            return {"estado": "esperando_tipo", "reply": ESTADOS["inicio"]}

        # Fases de filtrado
        if sesion["estado"] == "esperando_tipo":
            sesion["datos"]["tipo"] = limpio
            sesion["estado"] = "esperando_ingrediente"
            return {"estado": sesion["estado"], "reply": ESTADOS["tipo_elegido"]}

        if sesion["estado"] == "esperando_ingrediente":
            sesion["datos"]["ingrediente"] = limpio
            sesion["estado"] = "esperando_tiempo"
            return {"estado": sesion["estado"], "reply": ESTADOS["tiempo"]}

        if sesion["estado"] == "esperando_tiempo":
            sesion["datos"]["tiempo"] = limpio
            sesion["estado"] = "esperando_dieta"
            return {"estado": sesion["estado"], "reply": ESTADOS["ingrediente"]}

        if sesion["estado"] == "esperando_dieta":
            sesion["datos"]["dieta"] = limpio

            datos = sesion["datos"]
            busqueda = f"cocina {datos['tipo']}, ingrediente {datos['ingrediente']}, tiempo {datos['tiempo']}, dieta {datos['dieta']}"
            
            recetas, distances = self.recuperar_chunks(busqueda, top_k=top_k)

            # Valida que el prompt del usuario tenga sentido
            if not recetas or distances[0] > 1: 
                self.sesiones[session_id] = {"estado": "esperando_tipo", "datos": {}}
                return {
                    "estado": "escalado",
                    "reply": "No he podido entender tus preferencias o no encuentro nada remotamente parecido en nuestro recetario. Te paso con un agente humano.",
                    "escalated": True,
                    "top_k": top_k,
                    "retrieved_chunks": [],
                }

            # Lógica de Decisión
            if not recetas:
                self.sesiones[session_id] = {"estado": "esperando_tipo", "datos": {}}
                return {
                    "estado": "escalado",
                    "reply": escalar_a_humano(busqueda),
                    "escalated": True,
                    "top_k": top_k,
                    "retrieved_chunks": [],
                }

            # Construir el contexto y el Prompt personalizado para el Chef
            context = self.build_context(recetas)
            
            prompt_chef = (
                "Eres un Chef de un restaurante con estrellas Michelin. "
                "El cliente te ha pedido una receta con estas preferencias exactas:\n"
                f"- Tipo: {datos['tipo']}\n"
                f"- Ingrediente principal: {datos['ingrediente']}\n"
                f"- Tiempo disponible: {datos['tiempo']}\n"
                f"- Dieta: {datos['dieta']}\n\n"
                f"Aquí tienes las recetas disponibles en tu base de datos (RAG):\n{context}\n\n"
                "Instrucciones:\n"
                "1. Evalúa si las preferencias del cliente tienen sentido. Si son letras sin sentido (ej. 'aa') o incoherentes, responde textualmente: 'No tengo información suficiente para esta solicitud'.\n"
                "2. Si la petición tiene sentido, selecciona la mejor receta del contexto PERO NO EXPLIQUES tu proceso de selección. Ve directamente a la presentación elegante del plato.\n"
                "3. Escribe la receta COMPLETA (todos los ingredientes y todos los pasos). No la cortes ni la abrevies.\n"
                "4. Responde siempre en Español usando formato markdown.\n"
                "5. USA SOLO LA INFORMACIÓN DEL CONTEXTO. Si ninguna encaja lógicamente, responde: 'No tengo información suficiente para esta solicitud'."
                )
            if self.llm_service is None:
                respuesta_llm = f"[Modo Fallback - Sin LLM]\nBasado en tu búsqueda, encontré esto:\n{context}"
            else:
                respuesta_llm = self.llm_service.ask(prompt_chef)

            valida = es_respuesta_valida(respuesta_llm, context)
            
            self.sesiones[session_id] = {"estado": "esperando_tipo", "datos": {}}

            if not valida:
                return {
                    "estado": "escalado",
                    "reply": escalar_a_humano(busqueda),
                    "escalated": True,
                    "top_k": top_k,
                    "retrieved_chunks": recetas,
                    "distances": distances,
                }

            return {
                "estado": "resuelto",
                "reply": respuesta_llm,
                "escalated": False,
                "top_k": top_k,
                "retrieved_chunks": recetas,
                "distances": distances,
            }

        return {"estado": "desconocido", "reply": "Dime 'hola' para empezar de nuevo."}


def create_chef_agent_app() -> Flask:
    app = Flask(__name__)
    chef_agent = ChefRagAgent(build_llm_service())

    @app.get("/")
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True) or {}
        message = str(payload.get("message", "")).strip()
        session_id = str(payload.get("session_id", "default")).strip()
        top_k = int(payload.get("top_k", DEFAULT_TOP_K))
        if top_k <= 0:
            top_k = DEFAULT_TOP_K

        if not message:
            return jsonify({"ok": False, "reply": "Por favor, escribe un mensaje."}), 400

        resultado = chef_agent.run(session_id, message, top_k=top_k)
        return jsonify({
            "ok": True,
            "session_id": session_id,
            "estado": resultado["estado"],
            "reply": resultado["reply"],
            "debug": resultado
        })

    return app

def build_llm_service() -> LLMService | None:
    if not Config.LLM_ENABLED:
        return None
    return LLMService(
        LLMConfig(
            provider=Config.LLM_PROVIDER,
            api_key=Config.LLM_API_KEY,
            model=Config.LLM_MODEL,
            base_url=Config.LLM_BASE_URL,
            timeout=Config.LLM_TIMEOUT,
            max_tokens=Config.LLM_MAX_TOKENS,
        )
    )

app = create_chef_agent_app()

if __name__ == "__main__":
    port = int(os.environ.get("CHEF_AGENT_PORT", "5000"))
    app.run(debug=True, port=port)