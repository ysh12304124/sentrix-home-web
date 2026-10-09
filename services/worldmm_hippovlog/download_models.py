#!/usr/bin/env python3
"""Resumable model downloads directly to the 200 server."""
from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = {
    "qwen3vl4b": ("Qwen/Qwen3-VL-4B-Instruct", ROOT / "models" / "Qwen3-VL-4B-Instruct"),
    "qwen3embed4b": ("Qwen/Qwen3-Embedding-4B", ROOT / "models" / "Qwen3-Embedding-4B"),
    "vlm2vec": ("VLM2Vec/VLM2Vec-V2.0", ROOT / "models" / "VLM2Vec-V2.0"),
    "qwen2vl2b": ("Qwen/Qwen2-VL-2B-Instruct", ROOT / "models" / "Qwen2-VL-2B-Instruct"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()
    repo, destination = MODELS[args.model]
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
    from huggingface_hub import snapshot_download
    for attempt in range(1, 21):
        try:
            print(f"[download] {repo} attempt={attempt} destination={destination}", flush=True)
            path = snapshot_download(repo, local_dir=destination, max_workers=2)
            index_path = destination / "model.safetensors.index.json"
            if index_path.is_file():
                index = json.loads(index_path.read_text(encoding="utf-8"))
                shards = {destination / name for name in index["weight_map"].values()}
                total_bytes = sum(shard.stat().st_size for shard in shards if shard.is_file())
                expected = int(index.get("metadata", {}).get("total_size", 0))
                if len([shard for shard in shards if shard.is_file()]) != len(shards) or total_bytes < expected:
                    raise RuntimeError(f"weight shards incomplete: {total_bytes}/{expected} bytes across {len(shards)} shards")
            elif args.model == "vlm2vec":
                adapter = destination / "adapter_model.bin"
                config = destination / "adapter_config.json"
                if not config.is_file() or not adapter.is_file() or adapter.stat().st_size < 30_000_000:
                    raise RuntimeError("VLM2Vec LoRA adapter files are incomplete")
            elif not any(path.stat().st_size > 1_000_000 for path in destination.glob("*.safetensors")):
                raise RuntimeError("no complete safetensors model weights found")
            if list(destination.rglob("*.incomplete")):
                raise RuntimeError("partial Hugging Face transfers remain")
            print(f"[download complete] {repo} {path}", flush=True)
            return
        except Exception as error:
            print(f"[retry] {repo} attempt={attempt}: {type(error).__name__}: {error}", flush=True)
            if attempt == 20:
                raise
            time.sleep(min(120, 10 * attempt))


if __name__ == "__main__":
    main()
