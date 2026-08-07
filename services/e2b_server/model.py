"""E2B LoRA model loader for the local 153 GPU server.

Loads a base multimodal model + PEFT LoRA adapter, exposes async
generate() for the FastAPI server.  Two asyncio Locks serialise load
and inference to avoid OOM from concurrent GPU operations.
"""

import asyncio
import inspect
import os
import threading

import torch

from .ollama_shape import build_chat_messages

_DTYPE_TABLE = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
}


class E2BModel:
    def __init__(self, base_dir, adapter_dir, dtype="bf16", device_map="auto"):
        # type: (str, str, str, str) -> None
        self.base_dir = base_dir
        self.adapter_dir = adapter_dir
        self.dtype = dtype
        self.device_map = device_map
        self._model = None
        self._processor = None
        self._error = None  # type: str | None
        self._accepted_forward_keys = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()

    @property
    def is_loaded(self):
        # type: () -> bool
        return self._model is not None

    async def ensure_loaded(self):
        # type: () -> None
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._blocking_load)
            except Exception as exc:
                self._error = str(exc)
                raise

    def _blocking_load(self):
        # type: () -> None
        torch_dtype = getattr(torch, _DTYPE_TABLE.get(self.dtype, "float16"), torch.float16)

        # 1. Base model
        try:
            from transformers import AutoModelForMultimodalLM
            model = AutoModelForMultimodalLM.from_pretrained(
                self.base_dir,
                torch_dtype=torch_dtype,
                device_map=self.device_map,
                trust_remote_code=True,
            )
        except (ImportError, AttributeError):
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                self.base_dir,
                torch_dtype=torch_dtype,
                device_map=self.device_map,
                trust_remote_code=True,
            )

        # 2. LoRA adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, self.adapter_dir)

        # 3. Eval mode
        model.eval()

        try:
            signature = inspect.signature(model.forward)
        except (TypeError, ValueError):
            self._accepted_forward_keys = None
        else:
            if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
                self._accepted_forward_keys = None
            else:
                self._accepted_forward_keys = set(signature.parameters)

        # 4. Processor
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(self.base_dir, trust_remote_code=True)

        # Atomic assignment (Python GIL makes assignment of these references
        # thread-safe; the processor+model pair is always set together.)
        self._processor = processor
        self._model = model

    async def generate(self, prompt, images=None, **kwargs):
        # type: (str, list[str] | None, ...) -> str
        if self._model is None:
            await self.ensure_loaded()
        async with self._inference_lock:
            if self._model is None:
                # Double-check after acquiring the lock — a concurrent load
                # could have resolved the model.
                await self.ensure_loaded()
            # We know _model and _processor are set after ensure_loaded.
            assert self._model is not None
            assert self._processor is not None
            return self._generate_impl(self._model, self._processor, prompt, images, **kwargs)

    def _generate_impl(self, model, processor, prompt, images, **kwargs):
        # type: (...) -> str
        from PIL import Image
        from io import BytesIO
        import base64

        pil_images = []
        if images:
            for img in images:
                try:
                    data = base64.b64decode(img)
                    pil_images.append(Image.open(BytesIO(data)).convert("RGB"))
                except Exception:
                    continue

        max_new_tokens = kwargs.get("max_new_tokens", 512)
        temperature = kwargs.get("temperature", 0.0)
        do_sample = kwargs.get("do_sample", False)

        messages = build_chat_messages(prompt, pil_images)
        try:
            rendered_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        processor_kwargs = {
            "text": [rendered_prompt],
            "padding": True,
            "return_tensors": "pt",
        }
        if pil_images:
            # Gemma 4 expects one image list per text sample, not a flat list.
            processor_kwargs["images"] = [pil_images]
        inputs = processor(**processor_kwargs)

        unused = set(getattr(processor, "unused_input_names", ()) or ())
        inputs = {key: value for key, value in inputs.items() if value is not None and key not in unused}
        if self._accepted_forward_keys is not None:
            inputs = {key: value for key, value in inputs.items() if key in self._accepted_forward_keys}
        if "input_ids" not in inputs:
            raise RuntimeError("processor did not produce input_ids")

        # Move to the model's device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
            )

        # Decode only the newly generated tokens
        if isinstance(outputs, torch.Tensor):
            generated_ids = outputs[0]
            if "input_ids" in inputs:
                input_len = inputs["input_ids"].shape[-1]
                generated_ids = generated_ids[input_len:]
            if hasattr(processor, "batch_decode"):
                return processor.batch_decode([generated_ids], skip_special_tokens=True)[0].strip()
            return processor.decode(generated_ids, skip_special_tokens=True).strip()

        # Handle list output
        generated_ids = outputs[0]
        if hasattr(generated_ids, "shape") and "input_ids" in inputs:
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = generated_ids[input_len:]
        try:
            if hasattr(processor, "batch_decode"):
                return processor.batch_decode([generated_ids], skip_special_tokens=True)[0].strip()
            return processor.decode(generated_ids, skip_special_tokens=True).strip()
        except Exception:
            return str(generated_ids)

    async def unload(self):
        # type: () -> None
        async with self._load_lock:
            if self._model is not None:
                del self._model
                del self._processor
                self._model = None
                self._processor = None
                self._error = None
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
