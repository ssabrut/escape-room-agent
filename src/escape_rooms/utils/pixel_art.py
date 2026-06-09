"""Procedural pixel art sprite generator for WorldObjects.

Generates a 32×32 pixel art PNG (upscaled to 128×128 for display) for each
WorldObject, driven entirely by the object's id and description — no external
API calls, fully offline.

Strategy
--------
- Hash the object id to seed a deterministic RNG.
- Pick a body shape (silhouette) that hints at the object category
  (door, chest, key, lock, panel, paper/note, liquid, gear, default box).
- Fill the silhouette with a palette derived from the description keywords.
- Add a simple highlight/shadow pass for depth.
- Return the image as a base64-encoded PNG string.
"""

from __future__ import annotations

import base64
import hashlib
import io
import random
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from src.escape_rooms.state.schema import WorldObject

# ---------------------------------------------------------------------------
# Palette catalogue — (highlight, mid, shadow, outline)
# ---------------------------------------------------------------------------
_PALETTES: list[tuple[tuple[int, int, int], ...]] = [
    # warm wood
    ((210, 170, 110), (160, 110, 60), (100, 65, 30), (50, 30, 10)),
    # cold metal
    ((200, 215, 230), (140, 155, 170), (80, 95, 110), (30, 40, 55)),
    # aged parchment
    ((240, 230, 200), (190, 175, 140), (130, 110, 80), (60, 45, 20)),
    # mystic purple
    ((220, 180, 255), (160, 100, 220), (90, 40, 150), (40, 10, 80)),
    # poison green
    ((180, 240, 160), (100, 190, 80), (40, 120, 30), (10, 60, 5)),
    # fire orange
    ((255, 240, 130), (250, 160, 40), (200, 80, 10), (100, 30, 0)),
    # ocean blue
    ((180, 230, 255), (80, 160, 220), (20, 90, 160), (5, 30, 80)),
    # blood red
    ((255, 180, 160), (220, 80, 60), (150, 20, 10), (70, 0, 0)),
    # sandy stone
    ((240, 220, 190), (185, 160, 120), (120, 95, 60), (55, 35, 15)),
    # dark iron
    ((160, 160, 175), (100, 100, 115), (50, 50, 65), (15, 15, 25)),
]

# ---------------------------------------------------------------------------
# Shape definitions  (1 = body, 2 = highlight, 3 = detail/accent, 0 = empty)
# Each shape is 16×16 and will be tiled/scaled to fill the 32×32 canvas.
# ---------------------------------------------------------------------------

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
    "door": _SHAPE_DOOR,
    "key": _SHAPE_KEY,
    "lock": _SHAPE_LOCK,
    "chest": _SHAPE_CHEST,
    "scroll": _SHAPE_SCROLL,
    "bottle": _SHAPE_BOTTLE,
    "gear": _SHAPE_GEAR,
    "box": _SHAPE_BOX,
}

# Keywords that map to a specific shape
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


def _pick_palette(description: str, obj_id: str, rng: random.Random) -> tuple[tuple[int, int, int], ...]:
    lower = (description + " " + obj_id).lower()
    # Weight palettes by keyword hints
    keyword_palette: dict[str, int] = {
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


def _render_16x16(shape: list[str], palette: tuple[tuple[int, int, int], ...]) -> Image.Image:
    """Render a 16×16 RGBA image from a shape grid + 4-color palette."""
    highlight, mid, shadow, outline = palette
    color_map = {
        "0": (0, 0, 0, 0),          # transparent
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
    """Add subtle per-pixel brightness noise to break up flat fills."""
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


def _add_background(sprite: Image.Image, bg_color: tuple[int, int, int]) -> Image.Image:
    """Composite sprite onto a solid background square."""
    bg = Image.new("RGBA", sprite.size, (*bg_color, 255))
    bg.paste(sprite, mask=sprite)
    return bg


def generate_object_sprite(obj: "WorldObject", scale: int = 4) -> str:
    """Generate a pixel art PNG for *obj* and return it as a base64 string.

    Parameters
    ----------
    obj:
        The WorldObject to render.
    scale:
        Upscale multiplier applied after the 16×16 render (default 4 → 64×64).

    Returns
    -------
    str
        Base64-encoded PNG (no data-URI prefix).
    """
    seed = int(hashlib.md5(obj.id.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)

    shape = _pick_shape(obj.description, obj.id)
    palette = _pick_palette(obj.description, obj.id, rng)

    sprite = _render_16x16(shape, palette)
    sprite = _add_dither_noise(sprite, rng)

    # Pick a slightly dark background color from the mid-tone
    _, mid, _, _ = palette
    bg = tuple(max(0, c - 40) for c in mid)  # type: ignore[arg-type]
    img = _add_background(sprite, bg)  # type: ignore[arg-type]

    # Nearest-neighbour upscale for crisp pixels
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def generate_world_sprites(objects: list["WorldObject"]) -> dict[str, str]:
    """Return a mapping of object_id → base64 PNG for every object in *objects*."""
    return {obj.id: generate_object_sprite(obj) for obj in objects}
