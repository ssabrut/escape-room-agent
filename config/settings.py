import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()


class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    game_master_model: str = os.getenv("GAME_MASTER_MODEL", "llama3.2")
    game_extractor_model: str = os.getenv("GAME_EXTRACTOR_MODEL", "llama3.2")
    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))


@lru_cache(maxsize=1)
def get_extractor_llm() -> ChatOllama:
    s = Settings()
    return ChatOllama(
        model=s.game_extractor_model,
        base_url=s.ollama_base_url,
        temperature=s.temperature,
        reasoning=False,
    )


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    s = Settings()
    return ChatOllama(
        model=s.game_master_model,
        base_url=s.ollama_base_url,
        temperature=s.temperature,
        reasoning=False,
    )
