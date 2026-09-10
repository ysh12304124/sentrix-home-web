"""OpenAI-compatible local Qwen3-VL server with lazy load and idle unload."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
from pathlib import Path
import threading
import time
import traceback
import uuid

from aiohttp import web


class LocalQwenRuntime:
    def __init__(self, model_path: Path, served_name: str, idle_unload_seconds: float = 60.0):
        self.model_path = model_path
        self.served_name = served_name
        self.idle_unload_seconds = max(0.0, float(idle_unload_seconds))
        self.processor = None
        self.model = None
        self.torch = None
        self.load_error = ""
        self.loading = False
        self.last_used_at = 0.0
        self.load_lock = threading.Lock()
        self.generation_lock = threading.Lock()
        self.monitor_stop = threading.Event()

    @property
    def status(self) -> str:
        if self.model is not None:
            return "ready"
        if self.loading:
            return "loading"
        if self.load_error:
            return "error"
        return "unloaded"

    def load(self) -> None:
        with self.load_lock:
            if self.model is not None:
                return
            self.loading = True
            self.load_error = ""
            try:
                import torch
                from transformers import AutoModelForImageTextToText, AutoProcessor

                print(f"Loading local Qwen3-VL from {self.model_path}", flush=True)
                self.torch = torch
                if self.processor is None:
                    self.processor = AutoProcessor.from_pretrained(
                        str(self.model_path), trust_remote_code=True, local_files_only=True
                    )
                self.model = AutoModelForImageTextToText.from_pretrained(
                    str(self.model_path),
                    dtype="auto",
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                )
                self.model.eval()
                self.last_used_at = time.monotonic()
                print(f"Local model is ready: {self.served_name}", flush=True)
            except Exception as exc:  # pragma: no cover - hardware dependent
                self.load_error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            finally:
                self.loading = False

    def ensure_loaded(self) -> None:
        if self.model is None:
            self.load()
        if self.load_error:
            raise RuntimeError(self.load_error)
        if self.model is None:
            raise RuntimeError("local model did not load")

    def start_loading(self) -> None:
        if self.model is None and not self.loading:
            threading.Thread(target=self.load, name="qwen-model-loader", daemon=True).start()

    def unload(self, require_idle: bool = False) -> bool:
        acquired = self.generation_lock.acquire(blocking=not require_idle)
        if not acquired:
            return False
        try:
            if self.model is None:
                return False
            if require_idle:
                if self.idle_unload_seconds <= 0:
                    return False
                if time.monotonic() - self.last_used_at < self.idle_unload_seconds:
                    return False
            model = self.model
            self.model = None
            del model
            gc.collect()
            if self.torch is not None and self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
                try:
                    self.torch.cuda.ipc_collect()
                except Exception:
                    pass
            self.load_error = ""
            print(f"Local model unloaded: {self.served_name}", flush=True)
            return True
        finally:
            self.generation_lock.release()

    def start_idle_monitor(self) -> None:
        if self.idle_unload_seconds <= 0:
            return
        interval = min(5.0, max(1.0, self.idle_unload_seconds / 4))

        def monitor() -> None:
            while not self.monitor_stop.wait(interval):
                self.unload(require_idle=True)

        threading.Thread(target=monitor, name="qwen-idle-unloader", daemon=True).start()

    def model_device(self):
        try:
            return next(self.model.parameters()).device
        except Exception:
            return getattr(self.model, "device", "cuda")

    @staticmethod
    def normalize_messages(messages: list[dict]) -> list[dict]:
        normalized = []
        for message in messages:
            role = str(message.get("role") or "user")
            content = message.get("content", "")
            if isinstance(content, str):
                normalized.append({"role": role, "content": content})
                continue
            parts = []
            for part in content if isinstance(content, list) else []:
                if not isinstance(part, dict):
                    parts.append({"type": "text", "text": str(part)})
                    continue
                part_type = str(part.get("type") or "text")
                if part_type in {"text", "input_text"}:
                    parts.append({"type": "text", "text": str(part.get("text") or "")})
                elif part_type in {"image", "input_image", "image_url"}:
                    image = part.get("image") or part.get("image_url") or part.get("url")
                    if isinstance(image, dict):
                        image = image.get("url")
                    if image:
                        parts.append({"type": "image", "image": str(image)})
            normalized.append({"role": role, "content": parts})
        return normalized

    def generate(self, payload: dict) -> dict:
        from qwen_vl_utils import process_vision_info

        messages = self.normalize_messages(payload.get("messages") or [])
        if not messages:
            raise ValueError("messages must not be empty")
        max_new_tokens = max(1, min(int(payload.get("max_tokens") or 512), 4096))
        temperature = float(payload.get("temperature") or 0)
        top_p = float(payload.get("top_p") or 0.9)

        with self.generation_lock:
            self.ensure_loaded()
            try:
                rendered = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                processor_args = {"text": [rendered], "padding": True, "return_tensors": "pt"}
                if image_inputs:
                    processor_args["images"] = image_inputs
                if video_inputs:
                    processor_args["videos"] = video_inputs
                inputs = self.processor(**processor_args).to(self.model_device())
                generation_args = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
                if temperature > 0:
                    generation_args.update({"temperature": temperature, "top_p": top_p})
                with self.torch.inference_mode():
                    output_ids = self.model.generate(**inputs, **generation_args)
                prompt_tokens = int(inputs["input_ids"].shape[-1])
                generated = output_ids[:, prompt_tokens:]
                completion_tokens = int(generated.shape[-1])
                content = self.processor.batch_decode(
                    generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()
                del inputs, output_ids, generated
            finally:
                self.last_used_at = time.monotonic()
        return {"content": content, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


def create_app(runtime: LocalQwenRuntime) -> web.Application:
    routes = web.RouteTableDef()

    @routes.get("/")
    @routes.get("/health")
    async def health(_request):
        return web.json_response({
            "status": runtime.status,
            "model": runtime.served_name,
            "model_resident": runtime.model is not None,
            "idle_unload_seconds": runtime.idle_unload_seconds,
            "error": runtime.load_error or None,
        }, status=500 if runtime.load_error else 200)

    @routes.get("/v1/models")
    async def models(_request):
        return web.json_response({
            "object": "list",
            "data": [{"id": runtime.served_name, "object": "model", "created": int(time.time()), "owned_by": "local"}],
            "local_status": runtime.status,
            "model_resident": runtime.model is not None,
            "idle_unload_seconds": runtime.idle_unload_seconds,
        })

    @routes.post("/admin/model/load")
    async def load_model(_request):
        runtime.start_loading()
        return web.json_response({"status": runtime.status, "model_resident": runtime.model is not None}, status=200 if runtime.model is not None else 202)

    @routes.post("/admin/model/unload")
    async def unload_model(_request):
        unloaded = await asyncio.to_thread(runtime.unload)
        return web.json_response({"status": runtime.status, "model_resident": runtime.model is not None, "unloaded": unloaded})

    @routes.post("/v1/chat/completions")
    async def chat(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(runtime.generate, payload)
        except (ValueError, json.JSONDecodeError) as exc:
            return web.json_response({"error": {"message": str(exc), "type": "invalid_request_error"}}, status=400)
        except Exception as exc:
            return web.json_response({"error": {"message": str(exc), "type": "local_model_error"}}, status=503)

        created = int(time.time())
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        usage = {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        }
        if payload.get("stream"):
            response = web.StreamResponse(status=200, headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            })
            await response.prepare(request)
            chunks = [
                {"id": request_id, "object": "chat.completion.chunk", "created": created, "model": runtime.served_name, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
                {"id": request_id, "object": "chat.completion.chunk", "created": created, "model": runtime.served_name, "choices": [{"index": 0, "delta": {"content": result["content"]}, "finish_reason": None}]},
                {"id": request_id, "object": "chat.completion.chunk", "created": created, "model": runtime.served_name, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage},
            ]
            for chunk in chunks:
                await response.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response
        return web.json_response({
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": runtime.served_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": result["content"]}, "finish_reason": "stop"}],
            "usage": usage,
        })

    app = web.Application(client_max_size=128 * 1024 * 1024)
    app.add_routes(routes)

    async def cleanup(_app):
        runtime.monitor_stop.set()

    app.on_cleanup.append(cleanup)
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen3-VL-4B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11434)
    parser.add_argument("--idle-unload-seconds", type=float, default=float(os.getenv("LOCAL_QWEN_IDLE_UNLOAD_SECONDS", "60")))
    parser.add_argument("--eager-load", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        parser.error(f"model directory not found: {model_path}")
    runtime = LocalQwenRuntime(model_path, args.model_name, args.idle_unload_seconds)
    runtime.start_idle_monitor()
    if args.eager_load:
        runtime.start_loading()
    print(f"OpenAI-compatible Qwen server: http://{args.host}:{args.port}/v1", flush=True)
    web.run_app(create_app(runtime), host=args.host, port=args.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
