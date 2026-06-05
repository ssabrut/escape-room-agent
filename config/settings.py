import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    world_builder_model: str = os.getenv("WORLD_BUILDER_MODEL", "llama3.2")
    player_model: str = os.getenv(
        "PLAYER_MODEL", os.getenv("WORLD_BUILDER_MODEL", "llama3.2")
    )
    world_builder_temperature: float = float(os.getenv("WORLD_BUILDER_TEMPERATURE", "0.8"))
    player_temperature: float = float(os.getenv("PLAYER_TEMPERATURE", "0.3"))

    def __init__(self) -> None:
        # Hard-mode world generation: multi-room worlds with deep puzzle chains,
        # validated solvable before play. When off, the live game uses 2-room mode.
        # Read in
        # __init__ (not as class attrs) so main.py can set these env vars at
        # runtime via --hard and a freshly-built Settings() picks them up.
        self.hard_mode = _env_bool("HARD_MODE", False)
        self.num_rooms = int(os.getenv("NUM_ROOMS", "4"))
        self.chain_depth = int(os.getenv("CHAIN_DEPTH", "5"))
        # Regenerate up to this many times until the oracle confirms the world is
        # winnable (0 = no validation, accept the first build).
        self.gen_max_attempts = int(os.getenv("GEN_MAX_ATTEMPTS", "6"))


_ROLE_CONFIG = {
    "world_builder": lambda s: (s.world_builder_model, s.world_builder_temperature),
    "player": lambda s: (s.player_model, s.player_temperature),
}


@lru_cache(maxsize=4)
def get_llm(role: str = "world_builder") -> ChatOllama:
    s = Settings()
    resolver = _ROLE_CONFIG.get(role)
    if resolver is None:
        raise ValueError(
            f"Unknown LLM role: {role!r}. Expected one of {list(_ROLE_CONFIG)}"
        )
    model, temperature = resolver(s)
    return ChatOllama(
        model=model,
        base_url=s.ollama_base_url,
        temperature=temperature,
        num_predict=4096,
        reasoning=False,
        extra_body={"think": False},
    )
