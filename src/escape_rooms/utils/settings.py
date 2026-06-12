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
    # Comma-separated base URLs of additional Ollama instances (e.g. another Mac
    # on the LAN running the same BUILDER_MODEL). When set, independent LLM calls
    # within a generation step — per-room theming (puzzle_graph.apply_theming) and
    # the storyboard's beats/flavor passes (storyboard_builder) — are split
    # round-robin across the local Ollama instance and each of these, generated
    # concurrently. Mirrors SPRITE_WORKERS but talks directly to Ollama's HTTP API
    # (no extra worker process needed). OLLAMA_THEMING_WORKERS is accepted as a
    # legacy alias.
    ollama_workers: list[str] = [
        u.strip().rstrip("/")
        for u in os.getenv("OLLAMA_WORKERS", os.getenv("OLLAMA_THEMING_WORKERS", "")).split(",")
        if u.strip()
    ]
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
    # Storyboard generation runs once per world and benefits from a higher
    # temperature for creative variety; falls back to BUILDER_MODEL.
    storyboard_model: str = os.getenv(
        "STORYBOARD_MODEL", os.getenv("BUILDER_MODEL", "llama3.2")
    )
    storyboard_temperature: float = float(os.getenv("STORYBOARD_TEMPERATURE", "0.85"))
    # Narrator runs per-turn during gameplay — falls back to PLAYER_MODEL.
    narrator_model: str = os.getenv(
        "NARRATOR_MODEL", os.getenv("PLAYER_MODEL", os.getenv("BUILDER_MODEL", "llama3.2"))
    )
    narrator_temperature: float = float(os.getenv("NARRATOR_TEMPERATURE", "0.8"))
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
    "storyboard": lambda s: (s.storyboard_model, s.storyboard_temperature),
    "narrator": lambda s: (s.narrator_model, s.narrator_temperature),
}


@lru_cache(maxsize=16)
def get_llm(role: str = "game_master", base_url: str | None = None) -> ChatOllama:
    """Return a cached ChatOllama for `role`, optionally pointed at `base_url`
    instead of the default OLLAMA_BASE_URL (used to fan work out to additional
    Ollama instances — see Settings.ollama_workers)."""
    s = Settings()
    resolver = _ROLE_CONFIG.get(role)
    if resolver is None:
        raise ValueError(
            f"Unknown LLM role: {role!r}. Expected one of {list(_ROLE_CONFIG)}"
        )
    model, temperature = resolver(s)
    return ChatOllama(
        model=model,
        base_url=base_url or s.ollama_base_url,
        temperature=temperature,
        num_predict=s.ollama_num_predict,
        reasoning=False,
        extra_body={"think": False},
        keep_alive=s.ollama_keep_alive,
    )


def get_worker_llms(role: str, llm: ChatOllama | None = None) -> list[ChatOllama]:
    """Return `[llm (local)] + one ChatOllama per Settings.ollama_workers`.

    Lets a caller with N independent LLM calls round-robin them across the
    local Ollama instance and any additional LAN instances — the same
    fan-out pattern used by puzzle_graph.apply_theming and
    storyboard_builder's beats/flavor passes. `llm` defaults to `get_llm(role)`
    if not given.
    """
    if llm is None:
        llm = get_llm(role)
    worker_urls = Settings().ollama_workers
    return [llm] + [get_llm(role, base_url=url) for url in worker_urls]
