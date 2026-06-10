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
and starts `sprite_worker.py` on `0.0.0.0:8001`. Leave it running — it
advertises itself via Bonjour/mDNS and waits for sprite requests.

To only set up the env without starting the server:

```bash
./scripts/setup_worker.sh --setup
```

> First sprite request triggers a ~7GB SDXL model download — let the worker
> sit for a minute and consider sending one test request before a full run.

## 2. Main Mac — point at the worker

From the repo root on the main Mac, with no arguments to auto-discover the
worker on the LAN via Bonjour/mDNS:

```bash
./scripts/setup_main.sh
```

This writes `SPRITE_WORKERS=http://<discovered-ip>:8001` into `.env` and
checks the worker's `/health` endpoint. It exits with an error if no worker
is found — make sure `setup_worker.sh` is running on the other Mac first.

If auto-discovery doesn't find the worker (e.g. mDNS blocked on the
network), specify it manually instead:

```bash
./scripts/setup_main.sh <worker-ip-or-hostname> [port]
```

### Finding the worker's IP/hostname manually

On the **worker Mac**, run any of:

```bash
ipconfig getifaddr en0           # Wi-Fi/Ethernet LAN IP
ipconfig getifaddr en1           # try this if en0 has no address
scutil --get LocalHostName       # .local mDNS hostname (e.g. my-second-mac)
hostname                         # full hostname
```

`setup_worker.sh` prints the LAN IP at the end of setup. Use that IP (not
the `.local` hostname) with `setup_main.sh` — `curl`/`requests` don't
reliably resolve `.local` names even when `ping` does.

## 3. Run generation — main Mac only

Run your normal generation command (`python main.py ...` or the `/generate`
API) **only on the main Mac**. Sprite jobs are split round-robin between the
main Mac's local pipeline and the worker — both machines compute sprites in
parallel, but only the main Mac orchestrates the run.

To go back to local-only generation, remove or comment out `SPRITE_WORKERS`
in `.env`.
