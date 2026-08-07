"""Immutable Ollama-shaped protocol helpers for the E2B server.

All functions are pure (no side-effects, no I/O).  They parse/generate
JSON payloads that match Ollama /api/chat and /api/generate shapes so
that existing Gamma client code can talk to the E2B adapter without
modification.
"""

from typing import Any

JSON_TRAILING_HINT = "\n仅输出 JSON，不要包裹在代码块中。"


def build_chat_messages(prompt, pil_images):
    """Build native Gemma message blocks so the template emits image tokens."""
    content = [{"type": "image", "image": image} for image in pil_images]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def extract_prompt_and_images(payload):
    # type: (dict[str, Any]) -> tuple[str, list[str]]
    """Return (prompt_text, [base64_image_strings]) from an Ollama-shape
    /api/chat payload."""
    messages = payload.get("messages") or []
    prompt = ""
    images = []  # type: list[str]
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            prompt += content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    prompt += part
                elif isinstance(part, dict):
                    if part.get("type") == "text":
                        prompt += part.get("text", "")
                    elif part.get("type") == "image_url":
                        url = (part.get("image_url") or {}).get("url", "")
                        if url.startswith("data:"):
                            # "data:image/jpeg;base64,<b64>"
                            b64 = url.split(",", 1)[-1] if "," in url else url
                            images.append(b64)
        if msg.get("images"):
            images.extend(msg["images"])
        prompt += "\n"
    return prompt.strip(), images


def map_options(options):
    # type: (dict[str, Any] | None) -> dict[str, Any]
    """Translate Ollama-style options to generation kwargs."""
    if not options:
        return {}
    result = {}  # type: dict[str, Any]
    if options.get("num_predict"):
        result["max_new_tokens"] = int(options["num_predict"])
    if options.get("temperature") is not None:
        result["do_sample"] = float(options["temperature"]) > 0
        result["temperature"] = float(options["temperature"])
    return result


def build_chat_response(model, text):
    # type: (str, str) -> dict[str, Any]
    """Return an Ollama /api/chat response dict."""
    return {
        "model": model,
        "created_at": "",
        "message": {"role": "assistant", "content": text},
        "done": True,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": 0,
        "eval_duration": 0,
    }


def build_generate_response(model, text):
    # type: (str, str) -> dict[str, Any]
    """Return an Ollama /api/generate response dict."""
    return {
        "model": model,
        "created_at": "",
        "response": text,
        "done": True,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": 0,
        "eval_duration": 0,
    }
