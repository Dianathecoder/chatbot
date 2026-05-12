from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from flask import Flask, jsonify, render_template_string, request

from app.config import Config
from app.llm.llm_service import LLMConfig, LLMService


HTML_TEMPLATE = r"""
<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Agente matemático simple</title>
    <style>
      :root {
        --bg: #f5f7f8;
        --surface: #ffffff;
        --ink: #152025;
        --muted: #61717a;
        --line: #ccd6dc;
        --accent: #0f6b5f;
        --accent-dark: #084d45;
        --soft: #e8f3f0;
        --code-bg: #101719;
        --code-ink: #eef7f4;
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        color: var(--ink);
        background: var(--bg);
        font-family: Arial, Helvetica, sans-serif;
        letter-spacing: 0;
      }

      button,
      input {
        font: inherit;
        letter-spacing: 0;
      }

      main {
        width: min(1080px, 100%);
        min-height: 100vh;
        margin: 0 auto;
        padding: 18px;
      }

      h1,
      h2,
      p {
        margin-top: 0;
      }

      h1 {
        margin-bottom: 6px;
        font-size: 28px;
        line-height: 1.15;
      }

      h2 {
        margin-bottom: 10px;
        font-size: 20px;
      }

      p {
        line-height: 1.45;
      }

      .panel,
      .agent-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--surface);
      }

      .panel {
        padding: 16px;
      }

      .intro {
        margin-bottom: 14px;
      }

      .intro p {
        margin-bottom: 0;
        color: var(--muted);
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
      }

      button:hover {
        background: var(--accent-dark);
      }

      button.secondary {
        color: var(--accent);
        background: #ffffff;
      }

      button.secondary:hover {
        background: var(--soft);
      }

      .examples,
      .agent-grid {
        display: grid;
        gap: 12px;
      }

      .examples {
        grid-template-columns: repeat(4, minmax(0, 1fr));
        margin-bottom: 14px;
      }

      .agent-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .agent-card {
        min-height: 170px;
        padding: 14px;
      }

      .agent-card.active {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(15, 107, 95, 0.14);
      }

      .agent-card strong {
        display: block;
        margin-bottom: 8px;
        color: var(--accent-dark);
      }

      .agent-card p {
        margin-bottom: 0;
        color: var(--muted);
      }

      pre {
        overflow: auto;
        max-height: 320px;
        margin: 14px 0 0;
        border-radius: 6px;
        padding: 12px;
        color: var(--code-ink);
        background: var(--code-bg);
      }

      code {
        font-family: Consolas, "Courier New", monospace;
        font-size: 13px;
        line-height: 1.5;
      }

      @media (max-width: 820px) {
        .input-row,
        .examples,
        .agent-grid {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <main>
      <section class="panel intro">
        <h1>Agente matemático simple</h1>
        <p>Un agente resuelve suma, resta y multiplicación con reglas. Si la operación no es una de esas, la delega al agente LLM.</p>
      </section>

      <section class="panel">
        <div class="input-row">
          <input id="message-input" value="suma 2 y 3" aria-label="Operación">
          <button type="button" id="send-button">Procesar</button>
        </div>
        <div class="examples">
          <button type="button" class="secondary" data-message="suma 2 y 3">suma</button>
          <button type="button" class="secondary" data-message="resta 10 y 4">resta</button>
          <button type="button" class="secondary" data-message="multiplica 6 por 7">multiplica</button>
          <button type="button" class="secondary" data-message="divide 10 entre 2">delegar LLM</button>
        </div>

        <section class="agent-grid">
          <article class="agent-card" id="rule-agent">
            <strong>Agente 1: reglas aritméticas</strong>
            <p id="rule-copy">Esperando operación.</p>
          </article>
          <article class="agent-card" id="llm-agent">
            <strong>Agente 2: fallback LLM</strong>
            <p id="llm-copy">Solo actúa si no es suma, resta o multiplicación.</p>
          </article>
        </section>

        <pre><code id="result-json">{}</code></pre>
      </section>
    </main>

    <script>
      const input = document.querySelector("#message-input");
      const sendButton = document.querySelector("#send-button");
      const ruleAgent = document.querySelector("#rule-agent");
      const llmAgent = document.querySelector("#llm-agent");
      const ruleCopy = document.querySelector("#rule-copy");
      const llmCopy = document.querySelector("#llm-copy");
      const resultJson = document.querySelector("#result-json");

      async function send(message = input.value) {
        const response = await fetch("/api/calculate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message })
        });
        render(await response.json());
      }

      function render(result) {
        ruleAgent.classList.toggle("active", result.agent === "rules");
        llmAgent.classList.toggle("active", result.agent === "llm");
        ruleCopy.textContent = result.rule_agent || "Sin intervención.";
        llmCopy.textContent = result.llm_agent || "Sin intervención.";
        resultJson.textContent = JSON.stringify(result, null, 2);
      }

      sendButton.addEventListener("click", () => send());
      document.querySelectorAll("[data-message]").forEach((button) => {
        button.addEventListener("click", () => {
          input.value = button.dataset.message || "";
          send(input.value);
        });
      });

      send();
    </script>
  </body>
</html>
"""


@dataclass
class MathResult:
    operation: str
    a: float
    b: float
    result: float


class ArithmeticRuleAgent:
    def run(self, message: str) -> MathResult | None:
        lowered = message.lower().strip()
        numbers = self._numbers(lowered)
        if len(numbers) < 2:
            return None

        a, b = numbers[0], numbers[1]
        if any(token in lowered for token in ["suma", "sumar", "mas", "+"]):
            return MathResult("sumar", a, b, a + b)
        if any(token in lowered for token in ["resta", "restar", "menos", "-"]):
            return MathResult("restar", a, b, a - b)
        if any(token in lowered for token in ["multiplica", "multiplicar", "por", "x", "*"]):
            return MathResult("multiplicar", a, b, a * b)
        return None

    def _numbers(self, text: str) -> list[float]:
        return [float(match.replace(",", ".")) for match in re.findall(r"-?\d+(?:[\.,]\d+)?", text)]


class LlmFallbackAgent:
    def __init__(self, llm_service: LLMService | None):
        self.llm_service = llm_service

    def run(self, message: str) -> str:
        prompt = (
            "Resuelve esta operación matemática de forma breve. "
            "Si falta información, explica qué dato falta.\n\n"
            f"Operación: {message}"
        )
        if self.llm_service is None:
            return f"Delegaría al LLM con este prompt: {prompt}"
        return self.llm_service.ask(prompt)


def create_math_agent_app() -> Flask:
    app = Flask(__name__)
    rule_agent = ArithmeticRuleAgent()
    llm_agent = LlmFallbackAgent(build_llm_service())

    @app.get("/")
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.post("/api/calculate")
    def calculate():
        payload = request.get_json(silent=True) or {}
        message = str(payload.get("message", "")).strip()

        if not message:
            return jsonify({"ok": False, "error": "empty_message", "reply": "Falta la operación."}), 400

        rule_result = rule_agent.run(message)
        if rule_result is not None:
            return jsonify(
                {
                    "ok": True,
                    "agent": "rules",
                    "reply": f"Resultado: {format_number(rule_result.result)}",
                    "rule_agent": (
                        f"Detecta {rule_result.operation}: "
                        f"{format_number(rule_result.a)} y {format_number(rule_result.b)}."
                    ),
                    "llm_agent": "No se usa porque la operación está cubierta por reglas.",
                    "operation": rule_result.operation,
                    "numbers": [rule_result.a, rule_result.b],
                    "result": rule_result.result,
                }
            )

        llm_reply = llm_agent.run(message)
        return jsonify(
            {
                "ok": True,
                "agent": "llm",
                "reply": llm_reply,
                "rule_agent": "No reconoce suma, resta ni multiplicación con dos números.",
                "llm_agent": "Recibe la operación porque queda fuera de las reglas simples.",
                "delegated_message": message,
            }
        )

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


def format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value)


app = create_math_agent_app()


if __name__ == "__main__":
    port = int(os.environ.get("MATH_AGENT_PORT", "5057"))
    app.run(debug=True, port=port)