"""Sprite generator for WorldObjects using SD_PixelArt_SpriteSheet_Generator via diffusers.

Generates a 64×64 pixel-art PNG for each object using a locally-running
Stable Diffusion pipeline (Onodofthenorth/SD_PixelArt_SpriteSheet_Generator) on
Apple Silicon (MPS). The pipeline is lazy-loaded on the first call and kept in
memory for subsequent calls. Model weights are ~1GB and downloaded automatically
from HuggingFace on first use.

One-time setup:
    pip install torch torchvision diffusers transformers accelerate sentencepiece

Falls back to procedural pixel-art generation if diffusers/torch are not
installed or if the pipeline fails for any reason.
"""

from __future__ import annotations

import base64
import hashlib
import io
import random
from typing import TYPE_CHECKING

from PIL import Image
from src.escape_rooms.utils.logging import get_node_logger

if TYPE_CHECKING:
    from src.escape_rooms.state.schema import WorldObject

log = get_node_logger("pixel_art")

# ---------------------------------------------------------------------------
# SD pixel-art item/object backend
# ---------------------------------------------------------------------------

_SD_MODEL = "nerijs/pixel-art-xl"
_OUTPUT_SIZE = 64    # final sprite size after downscale
_GEN_SIZE = 512      # size to generate at (nearest-neighbour → pixel art look)
_SD_STEPS = 20       # SD 1.5 needs ~20 steps for good results

# Module-level lazy singleton — loaded once, reused across all calls.
_pipeline = None
_pipeline_error: Exception | None = None  # cached so we don't retry a broken env


def _get_pipeline():
    global _pipeline, _pipeline_error
    if _pipeline is not None:
        return _pipeline
    if _pipeline_error is not None:
        return None
    try:
        import torch
        from diffusers import DiffusionPipeline

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        log.info("Loading SD_PixelArt_SpriteSheet on {} ...", device)

        device_map = "cuda" if torch.cuda.is_available() else None
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        if device_map is None:
            pipe = pipe.to(device)
        pipe.load_lora_weights(_SD_MODEL)
        pipe.set_progress_bar_config(disable=True)
        _pipeline = pipe
        log.success("SD pipeline ready on {}", device)
        return _pipeline
    except Exception as exc:
        _pipeline_error = exc
        log.warning("SD unavailable ({}) — using procedural fallback", exc)
        return None


def _build_prompt(obj: "WorldObject") -> str:
    desc = obj.description.strip() if obj.description else obj.id.replace("_", " ")
    return (
        f"pixelsprite, {desc}, "
        "RPG game item icon, flat 2D top-down view, solid black background, "
        "16-bit pixel art, limited color palette, no text, crisp pixels, single object, "
        "centered, no other views, no variations, no grid, no spritesheet, no multiple objects, isolated icon"
    )

_NEGATIVE_PROMPT = (
    "human, person, character, humanoid, figure, body, face, limbs, "
    "blurry, photorealistic, 3d render, text, watermark, "
    "signature, low quality, grainy, multiple items"
)


def _generate_via_sd(obj: "WorldObject") -> str | None:
    pipe = _get_pipeline()
    if pipe is None:
        return None
    try:
        import torch

        seed = int(hashlib.md5(obj.id.encode()).hexdigest(), 16) % (2**32)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        result = pipe(
            prompt=_build_prompt(obj),
            negative_prompt=_NEGATIVE_PROMPT,
            width=_GEN_SIZE,
            height=_GEN_SIZE,
            num_inference_steps=_SD_STEPS,
            guidance_scale=7.5,
            generator=generator,
        )
        img: Image.Image = result.images[0]
        # Nearest-neighbour downscale gives the blocky pixel-art look
        img = img.resize((_OUTPUT_SIZE, _OUTPUT_SIZE), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        log.debug("SD sprite generated for {!r} ({} bytes b64)", obj.id, len(b64))
        return b64
    except Exception as exc:
        log.warning("SD generate failed for {!r}: {}", obj.id, exc)
        return None


# ---------------------------------------------------------------------------
# Procedural fallback (original pixel_art.py logic, kept intact)
# ---------------------------------------------------------------------------

_PALETTES: list[tuple[tuple[int, int, int], ...]] = [
    ((210, 170, 110), (160, 110, 60),  (100, 65, 30),  (50, 30, 10)),
    ((200, 215, 230), (140, 155, 170), (80, 95, 110),  (30, 40, 55)),
    ((240, 230, 200), (190, 175, 140), (130, 110, 80), (60, 45, 20)),
    ((220, 180, 255), (160, 100, 220), (90, 40, 150),  (40, 10, 80)),
    ((180, 240, 160), (100, 190, 80),  (40, 120, 30),  (10, 60, 5)),
    ((255, 240, 130), (250, 160, 40),  (200, 80, 10),  (100, 30, 0)),
    ((180, 230, 255), (80, 160, 220),  (20, 90, 160),  (5, 30, 80)),
    ((255, 180, 160), (220, 80, 60),   (150, 20, 10),  (70, 0, 0)),
    ((240, 220, 190), (185, 160, 120), (120, 95, 60),  (55, 35, 15)),
    ((160, 160, 175), (100, 100, 115), (50, 50, 65),   (15, 15, 25)),
]

_SHAPE_BOX = [
    "0000000000000000",
    "0111111111111110",
    "0122222222222210",
    "0122222222222210",
    "0122222222222210",
    "0122222222222210",
    "0122233332222210",
    "0122233332222210",
    "0122233332222210",
    "0122222222222210",
    "0122222222222210",
    "0122222222222210",
    "0122222222222210",
    "0111111111111110",
    "0000000000000000",
    "0000000000000000",
]
_SHAPE_DOOR = [
    "0000011111100000",
    "0000111111110000",
    "0001111111111000",
    "0011111111111100",
    "0011111111111100",
    "0011111111111100",
    "0011111111111100",
    "0011111111111100",
    "0011111111111100",
    "0011111111111100",
    "0011111111111100",
    "0011111111111100",
    "0011122222111100",
    "0011112222111100",
    "0001111111111000",
    "0000111111110000",
]
_SHAPE_KEY = [
    "0000000000000000",
    "0000001110000000",
    "0000011111000000",
    "0000111111100000",
    "0000111221100000",
    "0000111111100000",
    "0000011111000000",
    "0000001111000000",
    "0000000111000000",
    "0000000011100000",
    "0000000001130000",
    "0000000001130000",
    "0000000001110000",
    "0000000001110000",
    "0000000000000000",
    "0000000000000000",
]
_SHAPE_LOCK = [
    "0000011111100000",
    "0000111111110000",
    "0001110000011100",
    "0001100000001100",
    "0001100000001100",
    "0001110000011100",
    "0000111111110000",
    "0001111111111000",
    "0011111111111100",
    "0011111111111100",
    "0011111221111100",
    "0011111221111100",
    "0011111111111100",
    "0011111111111100",
    "0001111111111000",
    "0000111111110000",
]
_SHAPE_CHEST = [
    "0000011111100000",
    "0001111111111000",
    "0011222222221100",
    "0011222222221100",
    "0111222222222110",
    "1111222222222111",
    "1112222332222211",
    "1112222332222211",
    "1111111111111111",
    "1112222222222211",
    "1112222222222211",
    "1112222222222211",
    "1112222222222211",
    "1111222222221111",
    "0111111111111110",
    "0000000000000000",
]
_SHAPE_SCROLL = [
    "0000111111100000",
    "0001222222110000",
    "0011222222111000",
    "0111222222111100",
    "1112223322211110",
    "1112223322211110",
    "1112222222211110",
    "1112223322211110",
    "1112223322211110",
    "1112222222211110",
    "0111222222111100",
    "0011222222111000",
    "0001222222110000",
    "0000111111100000",
    "0000000000000000",
    "0000000000000000",
]
_SHAPE_BOTTLE = [
    "0000001111000000",
    "0000011221000000",
    "0000111221100000",
    "0000111111100000",
    "0001112221110000",
    "0011122222111000",
    "0111222222211100",
    "1112222222221110",
    "1112222222221110",
    "1112222222221110",
    "1112222222221110",
    "1112222222221110",
    "0111222222211100",
    "0011122222111000",
    "0001111111110000",
    "0000000000000000",
]
_SHAPE_GEAR = [
    "0000011111100000",
    "0001111111111000",
    "0011100000011100",
    "0111100000011110",
    "1111000011000111",
    "1110000111100011",
    "1100001111110001",
    "1100011111110001",
    "1100001111110001",
    "1110000111100011",
    "1111000011000111",
    "0111100000011110",
    "0011100000011100",
    "0001111111111000",
    "0000011111100000",
    "0000000000000000",
]

_SHAPES = {
    "door": _SHAPE_DOOR, "key": _SHAPE_KEY, "lock": _SHAPE_LOCK,
    "chest": _SHAPE_CHEST, "scroll": _SHAPE_SCROLL, "bottle": _SHAPE_BOTTLE,
    "gear": _SHAPE_GEAR, "box": _SHAPE_BOX,
}

_SHAPE_KEYWORDS: list[tuple[list[str], str]] = [
    (["door", "gate", "exit", "entrance", "hatch", "portal"], "door"),
    (["key", "keycard", "passkey"], "key"),
    (["lock", "padlock", "latch", "bolt"], "lock"),
    (["chest", "box", "crate", "safe", "cabinet", "drawer", "locker"], "chest"),
    (["scroll", "note", "letter", "paper", "document", "book", "journal", "diary"], "scroll"),
    (["bottle", "vial", "potion", "flask", "liquid", "jar", "canteen"], "bottle"),
    (["gear", "panel", "lever", "switch", "button", "control", "fuse", "circuit"], "gear"),
]


def _pick_shape(description: str, obj_id: str) -> list[str]:
    lower = (description + " " + obj_id).lower()
    for keywords, shape_name in _SHAPE_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return _SHAPES[shape_name]
    return _SHAPES["box"]


def _pick_palette(description: str, obj_id: str, rng: random.Random) -> tuple:
    lower = (description + " " + obj_id).lower()
    keyword_palette = {
        "gold|golden|treasure|ancient": 0,
        "metal|iron|steel|silver": 1,
        "paper|parchment|aged|old|dusty": 2,
        "magic|crystal|gem|glowing|rune": 3,
        "poison|slime|moss|plant|organic": 4,
        "fire|flame|hot|lava|torch": 5,
        "water|ocean|liquid|ink|glass": 6,
        "blood|rusty|stained|red": 7,
        "stone|rock|wall|concrete": 8,
        "dark|shadow|iron|black|grim": 9,
    }
    for patterns, idx in keyword_palette.items():
        if any(kw in lower for kw in patterns.split("|")):
            return _PALETTES[idx]
    return _PALETTES[rng.randint(0, len(_PALETTES) - 1)]


def _render_16x16(shape: list[str], palette: tuple) -> Image.Image:
    highlight, mid, shadow, outline = palette
    color_map = {
        "0": (0, 0, 0, 0),
        "1": (*outline, 255),
        "2": (*mid, 255),
        "3": (*highlight, 255),
    }
    img = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    pixels = img.load()
    for row_idx, row in enumerate(shape[:16]):
        for col_idx, cell in enumerate(row[:16]):
            pixels[col_idx, row_idx] = color_map.get(cell, (0, 0, 0, 0))
    return img


def _add_dither_noise(img: Image.Image, rng: random.Random, intensity: int = 12) -> Image.Image:
    pixels = img.load()
    for y in range(img.height):
        for x in range(img.width):
            r, g, b, a = pixels[x, y]
            if a == 0:
                continue
            delta = rng.randint(-intensity, intensity)
            pixels[x, y] = (
                max(0, min(255, r + delta)),
                max(0, min(255, g + delta)),
                max(0, min(255, b + delta)),
                a,
            )
    return img


def _add_background(sprite: Image.Image, bg_color: tuple) -> Image.Image:
    bg = Image.new("RGBA", sprite.size, (*bg_color, 255))
    bg.paste(sprite, mask=sprite)
    return bg


def _generate_procedural(obj: "WorldObject", scale: int = 4) -> str:
    seed = int(hashlib.md5(obj.id.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    shape = _pick_shape(obj.description, obj.id)
    palette = _pick_palette(obj.description, obj.id, rng)
    sprite = _render_16x16(shape, palette)
    sprite = _add_dither_noise(sprite, rng)
    _, mid, _, _ = palette
    bg = tuple(max(0, c - 40) for c in mid)  # type: ignore[arg-type]
    img = _add_background(sprite, bg)  # type: ignore[arg-type]
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_object_sprite(obj: "WorldObject") -> str:
    """Generate a sprite PNG for *obj* and return it as a base64 string.

    Uses SD_PixelArt_SpriteSheet on MPS when available; falls back to procedural generation.
    """
    result = _generate_via_sd(obj)
    if result is not None:
        return result
    return _generate_procedural(obj)


def generate_world_sprites(objects: list["WorldObject"]) -> dict[str, str]:
    """Return a mapping of object_id → base64 PNG for every object in *objects*."""
    log.info("Generating sprites for {} object(s)...", len(objects))
    sprites = {}
    for obj in objects:
        log.trace("  Generating sprite for {!r} — {!r}", obj.id, (obj.description or "")[:50])
        sprites[obj.id] = generate_object_sprite(obj)
    log.info("Sprite generation complete — {} sprite(s)", len(sprites))
    return sprites
