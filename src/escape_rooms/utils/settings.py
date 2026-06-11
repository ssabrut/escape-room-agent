import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _parse_keep_alive(val: str) -> "int | str":
    """Ollama keep_alive accepts an int (seconds; -1 = never unload) OR a duration
    string like "30m". A bare "-1" string is NOT a valid duration, so coerce plain
    integers to int and leave duration strings ("30m", "1h") untouched."""
    val = val.strip()
    try:
        return int(val)
    except ValueError:
        return val


class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    builder_model: str = os.getenv("BUILDER_MODEL", "llama3.2")
    player_model: str = os.getenv(
        "PLAYER_MODEL", os.getenv("BUILDER_MODEL", "llama3.2")
    )
    # Solver agent (agents.solver_agent) — its own model, independent of the
    # world generator and the in-game player. Falls back to BUILDER_MODEL.
    solver_model: str = os.getenv(
        "SOLVER_MODEL", os.getenv("BUILDER_MODEL", "llama3.2")
    )
    builder_temperature: float = float(os.getenv("BUILDER_TEMPERATURE", "0.8"))
    player_temperature: float = float(os.getenv("PLAYER_TEMPERATURE", "0.3"))
    # Solving wants determinism, not creativity — low temperature by default.
    solver_temperature: float = float(os.getenv("SOLVER_TEMPERATURE", "0.2"))
    # Inference tuning. keep_alive "-1" never unloads the model (no per-call reload
    # of a multi-GB model); num_predict caps tokens generated per call so a model
    # that runs away can't burn minutes on a single response.
    ollama_keep_alive = _parse_keep_alive(os.getenv("OLLAMA_KEEP_ALIVE", "-1"))
    ollama_num_predict: int = int(os.getenv("OLLAMA_NUM_PREDICT", "4096"))

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
    "game_master": lambda s: (s.builder_model, s.builder_temperature),
    "player": lambda s: (s.player_model, s.player_temperature),
    "solver": lambda s: (s.solver_model, s.solver_temperature),
}


@lru_cache(maxsize=4)
def get_llm(role: str = "game_master") -> ChatOllama:
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
        num_predict=s.ollama_num_predict,
        reasoning=False,
        extra_body={"think": False},
        keep_alive=s.ollama_keep_alive,
    )
