import os
from pathlib import Path


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on", "si"}


_load_env_file()


class Config:
    SECRET_KEY = "dev"
    JSON_SORT_KEYS = False

    CACHE_TYPE = "SimpleCache"
    CACHE_DEFAULT_TIMEOUT = 3600

    LLM_ENABLED = _env_bool("LLM_ENABLED", True)
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "lmstudio").strip().lower()
    LLM_BASE_URL = os.getenv("LLM_BASE_URL") or (
        "https://router.huggingface.co/v1"
        if LLM_PROVIDER == "huggingface"
        else os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    )
    LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "lm-studio"))
    LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "auto"))
    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "250"))