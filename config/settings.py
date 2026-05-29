import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()


class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    game_master_model: str = os.getenv("GAME_MASTER_MODEL", "llama3.2")
    player_model: str = os.getenv("PLAYER_MODEL", os.getenv("GAME_MASTER_MODEL", "llama3.2"))
    game_master_temperature: float = float(os.getenv("GAME_MASTER_TEMPERATURE", "0.8"))
    player_temperature: float = float(os.getenv("PLAYER_TEMPERATURE", "0.3"))


_ROLE_CONFIG = {
    "game_master": lambda s: (s.game_master_model, s.game_master_temperature),
    "player": lambda s: (s.player_model, s.player_temperature),
}


@lru_cache(maxsize=4)
def get_llm(role: str = "game_master") -> ChatOllama:
    s = Settings()
    resolver = _ROLE_CONFIG.get(role)
    if resolver is None:
        raise ValueError(f"Unknown LLM role: {role!r}. Expected one of {list(_ROLE_CONFIG)}")
    model, temperature = resolver(s)
    return ChatOllama(
        model=model,
        base_url=s.ollama_base_url,
        temperature=temperature,
        reasoning=False,
    )
