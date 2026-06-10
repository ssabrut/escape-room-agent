"""Sprite generator for WorldObjects using SDXL + the pixel-art-xl LoRA via diffusers.

Generates a 64×64 pixel-art PNG for each object using a locally-running
Stable Diffusion XL pipeline (stabilityai/stable-diffusion-xl-base-1.0 with the
nerijs/pixel-art-xl LoRA) on Apple Silicon (MPS). The pipeline is lazy-loaded on
the first call and kept in memory for subsequent calls. Model weights are
~7GB and downloaded automatically from HuggingFace on first use.

One-time setup:
    pip install torch torchvision diffusers transformers accelerate sentencepiece peft

Distributed generation:
    Sprite jobs are independent, so they can be fanned out across this
    machine plus one or more remote workers (see sprite_worker.py) running
    the same pipeline on other machines. Set SPRITE_WORKERS to a comma
    separated list of worker base URLs, e.g.:

        SPRITE_WORKERS=http://192.168.1.50:8001

    Jobs are split round-robin across the local pipeline and every
    configured remote worker, generated concurrently.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import requests
from diffusers.utils import logging as diffusers_logging
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from src.escape_rooms.utils.logging import get_node_logger

if TYPE_CHECKING:
    from src.escape_rooms.state.schema import WorldObject

load_dotenv()
log = get_node_logger("pixel_art")

# Silence the "input was truncated because CLIP can only handle ..." notice —
# our prompts are intentionally long and the truncation is expected/harmless.
diffusers_logging.set_verbosity_error()

# Silence diffusers' upcast_vae FutureWarning (deprecated kwarg, not actionable here).
warnings.filterwarnings(
    "ignore",
    message=r"`upcast_vae` is deprecated.*",
    category=FutureWarning,
)

# Comma-separated base URLs of remote sprite workers (see sprite_worker.py).
_REMOTE_WORKERS = [u.strip().rstrip("/") for u in os.getenv("SPRITE_WORKERS", "").split(",") if u.strip()]
_REMOTE_TIMEOUT = 300  # seconds — SDXL generation can take a while

# ---------------------------------------------------------------------------
# SD pixel-art item/object backend
# ---------------------------------------------------------------------------

_SD_MODEL = "nerijs/pixel-art-xl"
_OUTPUT_SIZE = 64    # final sprite size after downscale
_GEN_SIZE = 1024     # SDXL native resolution (nearest-neighbour → pixel art look)
_SD_STEPS = 25       # steps for good results with the pixel-art-xl LoRA

# Module-level lazy singleton — loaded once, reused across all calls.
_pipeline = None

# The pipeline's scheduler holds mutable per-call state (step index, sigmas),
# so concurrent pipe(...) calls from multiple threads corrupt each other's
# denoising loop. Serialize all local inference through this lock.
_pipeline_lock = threading.Lock()


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
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
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

    with _pipeline_lock:
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


def _generate_via_remote(worker_url: str, obj: "WorldObject") -> str:
    desc = obj.description.strip() if obj.description else obj.id.replace("_", " ")
    resp = requests.post(
        f"{worker_url}/sprite",
        json={"id": obj.id, "description": desc},
        timeout=_REMOTE_TIMEOUT,
    )
    resp.raise_for_status()
    b64 = resp.json()["sprite"]
    log.debug("Remote sprite generated for {!r} via {} ({} bytes b64)", obj.id, worker_url, len(b64))
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
    """Return a mapping of object_id → base64 PNG for every object in *objects*.

    If SPRITE_WORKERS lists one or more remote workers (see sprite_worker.py),
    the batch of objects is split round-robin between this machine's local
    pipeline and each remote worker, generated concurrently — each machine
    still generates its share sequentially on its own pipeline, but the
    shares run in parallel with each other.
    """
    total = len(objects)
    log.info("Generating sprites for {} object(s)...", total)

    if not _REMOTE_WORKERS or total <= 1:
        sprites = {}
        with tqdm(total=total, desc="Sprites", unit="obj") as pbar:
            for i, obj in enumerate(objects, start=1):
                log.trace("  Generating sprite for {!r} — {!r}", obj.id, (obj.description or "")[:50])
                start = time.monotonic()
                sprites[obj.id] = generate_object_sprite(obj)
                elapsed = time.monotonic() - start
                log.info("  [{}/{}] Generated sprite for {!r} in {:.1f}s", i, total, obj.id, elapsed)
                pbar.update(1)
        log.info("Sprite generation complete — {} sprite(s)", len(sprites))
        return sprites

    # One slot for the local pipeline, one per remote worker.
    slots: list = [generate_object_sprite] + [
        (lambda obj, _url=url: _generate_via_remote(_url, obj)) for url in _REMOTE_WORKERS
    ]
    log.info("Distributing sprite generation across {} machine(s) (1 local + {} remote)", len(slots), len(_REMOTE_WORKERS))

    sprites: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        batch_start = time.monotonic()
        futures = {
            pool.submit(slots[i % len(slots)], obj): obj
            for i, obj in enumerate(objects)
        }
        with tqdm(total=total, desc="Sprites", unit="obj") as pbar:
            for i, future in enumerate(futures, start=1):
                obj = futures[future]
                sprites[obj.id] = future.result()
                elapsed = time.monotonic() - batch_start
                log.info("  [{}/{}] Generated sprite for {!r} (batch elapsed {:.1f}s)", i, total, obj.id, elapsed)
                pbar.update(1)

    log.info("Sprite generation complete — {} sprite(s)", len(sprites))
    return sprites
