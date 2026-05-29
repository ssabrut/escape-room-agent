import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()


class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    game_master_model: str = os.getenv("GAME_MASTER_MODEL", "llama3.2")
    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))


@lru_cache(maxsize=1)
def get_llm(model: str = Settings().game_master_model) -> ChatOllama:
    s = Settings()
    return ChatOllama(
        model=model,
        base_url=s.ollama_base_url,
        temperature=s.temperature,
        reasoning=False,
    )
