from pathlib import Path

_ROOT = Path(__file__).parent


def load_prompt(node: str, name: str) -> str:
    return (_ROOT / node / f"{name}.txt").read_text(encoding="utf-8")
