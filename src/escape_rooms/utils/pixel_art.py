"""Sprite generator for WorldObjects using SDXL + the pixel-art-xl LoRA via diffusers.

Generates a 64×64 pixel-art PNG for each object using a locally-running
Stable Diffusion XL pipeline (stabilityai/stable-diffusion-xl-base-1.0 with the
nerijs/pixel-art-xl LoRA) on Apple Silicon (MPS). The pipeline is lazy-loaded on
the first call and kept in memory for subsequent calls. Model weights are
~7GB and downloaded automatically from HuggingFace on first use.

One-time setup:
    pip install torch torchvision diffusers transformers accelerate sentencepiece peft
"""

from __future__ import annotations

import base64
import hashlib
import io
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
_GEN_SIZE = 1024     # SDXL native resolution (nearest-neighbour → pixel art look)
_SD_STEPS = 25       # steps for good results with the pixel-art-xl LoRA

# Module-level lazy singleton — loaded once, reused across all calls.
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import torch
    from diffusers import DiffusionPipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Loading SDXL + pixel-art-xl LoRA on {} ...", device)

    device_map = "cuda" if torch.cuda.is_available() else None
    dtype = torch.float16 if device != "cpu" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        device_map=device_map,
    )
    if device_map is None:
        pipe = pipe.to(device)
    pipe.load_lora_weights(_SD_MODEL)
    pipe.set_progress_bar_config(disable=True)
    _pipeline = pipe
    log.success("SD pipeline ready on {}", device)
    return _pipeline


def _build_prompt(obj: "WorldObject") -> str:
    desc = obj.description.strip() if obj.description else obj.id.replace("_", " ")
    return (
        f"pixelsprite, {desc}, "
        "RPG game item icon, flat 2D top-down view, transparent background, "
        "16-bit pixel art, limited color palette, no text, crisp pixels, single object, "
        "centered, no other views, no variations, no grid, no spritesheet, "
        "no multiple objects, isolated icon"
    )

_NEGATIVE_PROMPT = (
    "human, person, character, humanoid, figure, body, face, limbs, "
    "blurry, photorealistic, 3d render, text, watermark, "
    "signature, low quality, grainy, multiple items"
)


def _generate_via_sd(obj: "WorldObject") -> str:
    import torch

    pipe = _get_pipeline()
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_object_sprite(obj: "WorldObject") -> str:
    """Generate a sprite PNG for *obj* and return it as a base64 string.

    Uses SDXL + pixel-art-xl LoRA.
    """
    return _generate_via_sd(obj)


def generate_world_sprites(objects: list["WorldObject"]) -> dict[str, str]:
    """Return a mapping of object_id → base64 PNG for every object in *objects*."""
    log.info("Generating sprites for {} object(s)...", len(objects))
    sprites = {}
    for obj in objects:
        log.trace("  Generating sprite for {!r} — {!r}", obj.id, (obj.description or "")[:50])
        sprites[obj.id] = generate_object_sprite(obj)
    log.info("Sprite generation complete — {} sprite(s)", len(sprites))
    return sprites
