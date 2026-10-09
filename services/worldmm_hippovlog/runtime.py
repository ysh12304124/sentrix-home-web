"""Compatibility layer for the unmodified WorldMM source on one RTX 3090."""
from __future__ import annotations

import os
import sys
import importlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def dataset_root() -> Path:
    return Path(os.environ.get(
        "HIPPOVLOG_DATASET_ROOT", "/ssd/sscy/datasets/HippoVlog-svd-worldmm-20260928"
    )).resolve()


def worldmm_root() -> Path:
    return Path(os.environ.get("WORLDMM_SOURCE_ROOT", str(dataset_root() / "source" / "WorldMM"))).resolve()


def add_worldmm_paths() -> None:
    source = worldmm_root()
    for path in (source / "src", source / "preprocess", source / "preprocess" / "episodic_memory"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def configure_worldmm() -> None:
    """Retain WorldMM's model interfaces while using the local single GPU."""
    add_worldmm_paths()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access the RTX 3090; check the CUDA environment")
    # Decord's native library can deadlock CUDA's first lazy initialization on
    # this host. Initialize CUDA before importing WorldMM's Qwen wrapper,
    # which imports decord at module import time.
    torch.cuda.init()
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from worldmm.llm import qwen3vl

    model_path = Path(os.environ.get(
        "WORLDMM_QWEN_MODEL_PATH",
        str(Path(__file__).resolve().parent / "models" / "Qwen3-VL-4B-Instruct"),
    )).resolve()
    if not (model_path / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"Qwen3-VL-4B model is incomplete: {model_path}")
    qwen3vl.MODEL_DICT["qwen3vl-4b"] = str(model_path)

    def init_single_gpu(self) -> None:
        self.kwargs.setdefault("max_new_tokens", 4096)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        self.model = self.model.to("cuda:0")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    qwen3vl.Qwen3VLModel._init_model = init_single_gpu

    # The official evaluator creates separate retriever and responder wrappers.
    # They use the same checkpoint here, so keep one weight copy in GPU memory.
    from worldmm.llm.llm_wrapper import LLMModel
    original_init = LLMModel._init_model
    shared_models = {}

    def init_shared(self, **kwargs):
        if self.provider != "qwen3vl":
            return original_init(self, **kwargs)
        key = self.model_name
        if key not in shared_models:
            shared_models[key] = original_init(self, **kwargs)
        elif kwargs.get("fps") is not None:
            shared_models[key].fps = kwargs["fps"]
        return shared_models[key]

    LLMModel._init_model = init_shared

    # WorldMM's embedding wrapper requests flash-attention 2, which is not
    # installed on this host. SDPA uses the same model and embeddings.
    from worldmm.embedding import qwen3_embedding

    def init_text_embedding(self, model_name="Qwen/Qwen3-Embedding-4B", device="auto") -> None:
        from sentence_transformers import SentenceTransformer
        local = os.environ.get("WORLDMM_TEXT_MODEL_PATH", model_name)
        device = os.environ.get("WORLDMM_TEXT_MODEL_DEVICE", device)
        self.model_name = local
        self.device = device
        self.model = SentenceTransformer(
            local,
            model_kwargs={"attn_implementation": "sdpa", "torch_dtype": torch.bfloat16},
            tokenizer_kwargs={"padding_side": "left"},
            device="cuda:0" if device in {"cuda", "auto"} else device,
        )

    qwen3_embedding.Qwen3EmbeddingModel.__init__ = init_text_embedding
    original_encode_text = qwen3_embedding.Qwen3EmbeddingModel.encode_text

    def encode_text_bounded(self, texts, batch_size=256):
        cap = os.environ.get("WORLDMM_TEXT_BATCH_SIZE")
        if cap:
            batch_size = min(batch_size, int(cap))
        elif self.device == "cpu":
            batch_size = min(batch_size, 8)
        return original_encode_text(self, texts, batch_size=batch_size)

    qwen3_embedding.Qwen3EmbeddingModel.encode_text = encode_text_bounded

    from worldmm.embedding.embedding_wrapper import EmbeddingModel
    original_embedding_init = EmbeddingModel.__init__

    def init_embedding_wrapper(self, text_model_name="Qwen/Qwen3-Embedding-4B",
                               vis_model_name="VLM2Vec/VLM2Vec-V2.0", device="cuda"):
        model_root = Path(__file__).resolve().parent / "models"
        text_name = os.environ.get("WORLDMM_TEXT_MODEL_PATH", str(model_root / "Qwen3-Embedding-4B"))
        vis_name = os.environ.get("WORLDMM_VIS_MODEL_PATH", str(model_root / "VLM2Vec-V2.0"))
        original_embedding_init(self, text_model_name=text_name, vis_model_name=vis_name, device=device)

    EmbeddingModel.__init__ = init_embedding_wrapper


def limit_worldmm_generation_threads() -> None:
    """Keep local GPU generation serial inside WorldMM's parallel batches."""
    modules = (
        "worldmm.memory.episodic.openie",
        "worldmm.memory.semantic.semantic_extraction",
        "worldmm.memory.semantic.semantic_consolidation",
        "worldmm.memory.episodic.multiscale",
        "worldmm.memory.episodic.gen_multiscale",
    )

    def single_worker_pool(*args, **kwargs):
        kwargs["max_workers"] = 1
        return ThreadPoolExecutor(*args, **kwargs)

    for module_name in modules:
        module = importlib.import_module(module_name)
        if hasattr(module, "ThreadPoolExecutor"):
            module.ThreadPoolExecutor = single_worker_pool


def configure_visual_sdpa() -> None:
    """Use SDPA for WorldMM's VLM2Vec loader on this flash-attn-free host."""
    add_worldmm_paths()
    model_module = importlib.import_module("worldmm.embedding.VLM2Vec.src.model.model")
    visual_module = importlib.import_module("worldmm.embedding.vlm2vecv2")

    def load_visual_with_local_base(self):
        adapter_path = Path(self.model_name).resolve()
        base_path = Path(os.environ.get(
            "WORLDMM_VIS_BASE_MODEL_PATH",
            str(Path(__file__).resolve().parent / "models" / "Qwen2-VL-2B-Instruct"),
        )).resolve()
        if not (adapter_path / "adapter_model.bin").is_file():
            raise FileNotFoundError(f"VLM2Vec adapter missing: {adapter_path}")
        if not (base_path / "config.json").is_file():
            raise FileNotFoundError(f"Qwen2-VL-2B base missing: {base_path}")
        model_args = visual_module.ModelArguments(
            model_name=str(base_path), checkpoint_path=str(adapter_path),
            pooling=self.pooling, normalize=self.normalize,
            model_backbone="qwen2_vl", lora=True,
        )
        self.processor = visual_module.load_processor(model_args, visual_module.DataArguments())
        self.model = visual_module.MMEBModel.load(model_args, is_trainable=False)
        self.model = self.model.to(self.device, dtype=model_module.torch.bfloat16)
        self.model.eval()

    visual_module.VLM2VecV2EmbeddingModel._load_model = load_visual_with_local_base

    @classmethod
    def load_sdpa(cls, model_args, is_trainable=True, **kwargs):
        model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
        config = model_module.AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
            model_args.model_backbone = model_module.get_backbone_name(
                hf_config=config, model_type=model_args.model_type
            )
        model_module.print_master(f"Loading backbone [{model_args.model_backbone}] from {model_name_or_path}")
        if model_args.model_backbone in {model_module.QWEN2_VL, model_module.QWEN2_VL_TOKENSELECTION}:
            config = model_module.AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config._attn_implementation = "sdpa"
            config.vision_config._attn_implementation = "sdpa"
            base_model = model_module.backbone2model[model_args.model_backbone].from_pretrained(
                model_args.model_name,
                torch_dtype=model_module.torch.bfloat16,
                low_cpu_mem_usage=True,
                config=config,
            )
        else:
            config = model_module.AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_name_or_path, **kwargs, config=config,
                torch_dtype=model_module.torch.bfloat16, trust_remote_code=True,
            )
        if model_args.lora:
            model_module.print_master(f"Loading LoRA from {model_name_or_path}")
            lora_config = model_module.LoraConfig.from_pretrained(model_name_or_path)
            lora_model = model_module.PeftModel.from_pretrained(
                base_model, model_name_or_path, config=lora_config, is_trainable=is_trainable
            )
            lora_model.load_adapter(model_name_or_path, lora_model.active_adapter, is_trainable=is_trainable)
            if not is_trainable:
                lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model, pooling=model_args.pooling,
                normalize=model_args.normalize, temperature=model_args.temperature,
            )
        else:
            model = cls(
                encoder=base_model, pooling=model_args.pooling,
                normalize=model_args.normalize, temperature=model_args.temperature,
            )
        model.model_backbone = model_args.model_backbone
        return model

    model_module.MMEBModel.load = load_sdpa
