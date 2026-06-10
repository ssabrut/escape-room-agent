# Distributed sprite generation (2 Macs over LAN)

Splits pixel-art sprite generation across two Macs to roughly halve total
generation time. One Mac (**main**) runs world generation as usual; the
other (**worker**) runs a small server that generates its share of the
sprites over the LAN.

## 1. Worker Mac — start the sprite worker

Make sure this repo is present on the worker Mac (clone or copy it), then
from its root:

```bash
./scripts/setup_worker.sh
```

This creates/uses the `escape-rooms` conda env, installs `requirements.txt`,
and starts `sprite_worker.py` on `0.0.0.0:8001`. Leave it running — it just
waits for sprite requests.

To only set up the env without starting the server:

```bash
./scripts/setup_worker.sh --setup
```

The script prints the worker's `.local` hostname, e.g.:

```
Main Mac should set: SPRITE_WORKERS=http://my-second-mac.local:8001
```

> First sprite request triggers a ~7GB SDXL model download — let the worker
> sit for a minute and consider sending one test request before a full run.

## 2. Main Mac — point at the worker

From the repo root on the main Mac:

```bash
./scripts/setup_main.sh my-second-mac.local
```

This writes `SPRITE_WORKERS=http://my-second-mac.local:8001` into `.env` and
checks the worker's `/health` endpoint. It exits with an error if the worker
isn't reachable yet.

## 3. Run generation — main Mac only

Run your normal generation command (`python main.py ...` or the `/generate`
API) **only on the main Mac**. Sprite jobs are split round-robin between the
main Mac's local pipeline and the worker — both machines compute sprites in
parallel, but only the main Mac orchestrates the run.

To go back to local-only generation, remove or comment out `SPRITE_WORKERS`
in `.env`.
