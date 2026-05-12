from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    api_key: str
    model: str
    timeout: float
    max_tokens: int


class LLMService:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider = config.provider.strip().lower()
        self.base_url = config.base_url.rstrip("/")
        self.api_key = config.api_key or self._default_api_key()
        self.model = config.model
        self.timeout = config.timeout
        self.max_tokens = config.max_tokens

    def ask(self, prompt: str) -> str:
        try:
            model = self._resolve_model()
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Eres un Chef virtual servicial, sofisticado y amable. "
                                "Te especializas en presentar recetas culinarias de manera atractiva. "
                                "Responde siempre en español y utiliza el formato markdown para que se lea bonito."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    "temperature": 0.5,
                    "max_tokens": self.max_tokens,
                    "stream": False,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return self._extract_content(data)
        except Exception as e:
            return f"{self._provider_error_prefix()} Detalle: {e}"

    def status(self) -> dict[str, Any]:
        base_status = {
            "provider": self.provider,
            "base_url": self.base_url,
            "configured_model": self.model,
            "selected_model": None,
            "available_models": [],
            "models_endpoint_ok": False,
            "api_key_configured": bool(self.api_key),
        }

        if self.provider == "huggingface" and self._uses_auto_model():
            return {
                **base_status,
                "ok": False,
                "error": "Hugging Face requiere fijar LLM_MODEL; no uses auto.",
            }

        try:
            models = self._list_models()
            return {
                **base_status,
                "ok": True,
                "selected_model": self._resolve_model(models),
                "available_models": models,
                "models_endpoint_ok": True,
                "error": None,
            }
        except Exception as e:
            if self.provider == "huggingface":
                return {
                    **base_status,
                    "ok": True,
                    "selected_model": self.model,
                    "error": (
                        "No se pudo listar modelos desde Hugging Face, pero la configuración es usable "
                        "si LLM_API_KEY y LLM_MODEL son correctos."
                    ),
                    "details": str(e),
                }

            return {
                **base_status,
                "ok": False,
                "error": str(e),
            }

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _resolve_model(self, models: list[str] | None = None) -> str:
        if not self._uses_auto_model():
            return self.model

        if self.provider == "huggingface":
            raise RuntimeError("Hugging Face requiere LLM_MODEL explícito; LLM_MODEL=auto no es válido.")

        models = models if models is not None else self._list_models()
        if not models:
            raise RuntimeError("El proveedor no devolvió modelos en /v1/models.")

        return models[0]

    def _list_models(self) -> list[str]:
        response = requests.get(
            f"{self.base_url}/models",
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return [model["id"] for model in data.get("data", []) if "id" in model]

    def _extract_content(self, data: dict[str, Any]) -> str:
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Respuesta LLM inesperada: {data}") from exc

    def _default_api_key(self) -> str:
        if self.provider == "lmstudio":
            return "lm-studio"
        return ""

    def _uses_auto_model(self) -> bool:
        return not self.model or self.model.strip().lower() == "auto"

    def _provider_error_prefix(self) -> str:
        if self.provider == "lmstudio":
            return (
                "Error conectando con LM Studio. Comprueba que el servidor local está arrancado, "
                "que hay un modelo cargado y que LLM_BASE_URL/LLM_MODEL son correctos."
            )

        if self.provider == "huggingface":
            return (
                "Error conectando con Hugging Face. Comprueba LLM_API_KEY, créditos disponibles, "
                "LLM_MODEL y LLM_BASE_URL."
            )

        return "Error conectando con el proveedor LLM compatible con OpenAI."