from __future__ import annotations

import os

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from api.routers import generate

API_KEY = os.getenv("API_KEY", "")

app = FastAPI(title="Escape Rooms API", version="1.0.0")


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
    return await call_next(request)


app.include_router(generate.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
