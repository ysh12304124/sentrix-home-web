"""FastAPI server that wraps the E2B LoRA model behind an Ollama-compatible
HTTP API.  Clients (GammaClient) send /api/chat payloads in Ollama JSON
shape; we translate, run inference, and return Ollama-shaped responses.
"""

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model import E2BModel
from .ollama_shape import (
    build_chat_response,
    build_generate_response,
    extract_prompt_and_images,
    map_options,
)

BASE_MODEL = os.getenv("E2B_BASE_MODEL", "/home/asus/models/gemma-4-E2B-it")
ADAPTER = os.getenv("E2B_ADAPTER", "/home/asus/models/gemma-4-e2b-lora-v2/artifacts/lora/V2_student_step47")
DTYPE = os.getenv("E2B_DTYPE", "bf16")
DEVICE_MAP = os.getenv("E2B_DEVICE_MAP", "auto")

model = E2BModel(BASE_MODEL, ADAPTER, dtype=DTYPE, device_map=DEVICE_MAP)

app = FastAPI(title="Sentrix E2B Server", version="0.1.0")


class ChatBody(BaseModel):
    model: str
    messages: list[dict]
    stream: bool = False
    format: str | None = None
    options: dict | None = None
    keep_alive: int | str | None = None


class GenerateBody(BaseModel):
    model: str
    prompt: str
    stream: bool = False
    options: dict | None = None


@app.get("/api/health")
async def health():
    return {
        "status": "ok" if model.is_loaded or model._error is None else "degraded",
        "model": os.path.basename(BASE_MODEL),
        "adapter": os.path.basename(ADAPTER),
        "dtype": DTYPE,
        "loaded": model.is_loaded,
        "error": model._error,
    }


@app.post("/api/chat")
async def chat(body: ChatBody):
    if not model.is_loaded and not body.model:
        raise HTTPException(status_code=503, detail="model not loaded")
    prompt_text, images = extract_prompt_and_images(body.model_dump())
    gen_kwargs = map_options(body.options)
    try:
        text = await model.generate(prompt_text, images, **gen_kwargs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return build_chat_response(body.model, text)


@app.post("/api/generate")
async def generate(body: GenerateBody):
    prompt_text, images = extract_prompt_and_images({"messages": [{"role": "user", "content": body.prompt}]})
    gen_kwargs = map_options(body.options)
    try:
        text = await model.generate(prompt_text, images, **gen_kwargs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return build_generate_response(body.model, text)


@app.post("/api/embeddings")
async def embeddings():
    raise HTTPException(status_code=501, detail="E2B server does not support embeddings")


@app.post("/admin/load")
async def admin_load():
    try:
        await model.ensure_loaded()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"loaded": True}


@app.post("/admin/unload")
async def admin_unload():
    await model.unload()
    return {"loaded": False}
