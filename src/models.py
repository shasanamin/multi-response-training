from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    family: str
    chat_model: bool
    trust_remote_code: bool = True
    auto_model_class: str = "causal_lm"
    prefer_processor: bool = False


class TextPreprocessor:
    def __init__(self, *, tokenizer: Any, processor: Any | None = None) -> None:
        self.tokenizer = tokenizer
        self.processor = processor

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.processor is not None:
            try:
                return self.processor(*args, **kwargs)
            except TypeError:
                pass
        return self.tokenizer(*args, **kwargs)

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(*args, **kwargs)
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(*args, **kwargs)
        if self.processor is not None and hasattr(self.processor, "decode"):
            return self.processor.decode(*args, **kwargs)
        raise AttributeError("The loaded text preprocessor does not implement decode().")

    def batch_decode(self, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self.tokenizer, "batch_decode"):
            return self.tokenizer.batch_decode(*args, **kwargs)
        if self.processor is not None and hasattr(self.processor, "batch_decode"):
            return self.processor.batch_decode(*args, **kwargs)
        raise AttributeError("The loaded text preprocessor does not implement batch_decode().")

    def __getattr__(self, name: str) -> Any:
        if name == "tokenizer":
            raise AttributeError(name)
        if hasattr(self.tokenizer, name):
            return getattr(self.tokenizer, name)
        if self.processor is not None and hasattr(self.processor, name):
            return getattr(self.processor, name)
        raise AttributeError(name)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "llama32_1b": ModelSpec(
        key="llama32_1b",
        hf_id="meta-llama/Llama-3.2-1B",
        family="llama",
        chat_model=False,
    ),
    "llama32_1b_instruct": ModelSpec(
        key="llama32_1b_instruct",
        hf_id="meta-llama/Llama-3.2-1B-Instruct",
        family="llama",
        chat_model=True,
    ),
    "llama32_3b": ModelSpec(
        key="llama32_3b",
        hf_id="meta-llama/Llama-3.2-3B",
        family="llama",
        chat_model=False,
    ),
    "llama32_3b_instruct": ModelSpec(
        key="llama32_3b_instruct",
        hf_id="meta-llama/Llama-3.2-3B-Instruct",
        family="llama",
        chat_model=True,
    ),
    "llama31_8b": ModelSpec(
        key="llama31_8b",
        hf_id="meta-llama/Llama-3.1-8B",
        family="llama",
        chat_model=False,
    ),
    "llama31_8b_instruct": ModelSpec(
        key="llama31_8b_instruct",
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        family="llama",
        chat_model=True,
    ),
    "gemma3_1b_it": ModelSpec(
        key="gemma3_1b_it",
        hf_id="google/gemma-3-1b-it",
        family="gemma3",
        chat_model=True,
    ),
    "gemma3_4b_it": ModelSpec(
        key="gemma3_4b_it",
        hf_id="google/gemma-3-4b-it",
        family="gemma3",
        chat_model=True,
    ),
    "gemma3_4b_pt": ModelSpec(
        key="gemma3_4b_pt",
        hf_id="google/gemma-3-4b-pt",
        family="gemma3",
        chat_model=False,
    ),
    "qwen35_2b": ModelSpec(
        key="qwen35_2b",
        hf_id="Qwen/Qwen3.5-2B",
        family="qwen35",
        chat_model=True,
        auto_model_class="image_text_to_text",
        prefer_processor=True,
    ),
    "qwen35_4b": ModelSpec(
        key="qwen35_4b",
        hf_id="Qwen/Qwen3.5-4B",
        family="qwen35",
        chat_model=True,
        auto_model_class="image_text_to_text",
        prefer_processor=True,
    ),
    "gemma4_e2b_it": ModelSpec(
        key="gemma4_e2b_it",
        hf_id="google/gemma-4-E2B-it",
        family="gemma4",
        chat_model=True,
        auto_model_class="image_text_to_text",
        prefer_processor=True,
    ),
    "gemma4_e4b_it": ModelSpec(
        key="gemma4_e4b_it",
        hf_id="google/gemma-4-E4B-it",
        family="gemma4",
        chat_model=True,
        auto_model_class="image_text_to_text",
        prefer_processor=True,
    ),
    "ministral3_3b_instruct": ModelSpec(
        key="ministral3_3b_instruct",
        hf_id="mistralai/Ministral-3-3B-Instruct-2512",
        family="ministral3",
        chat_model=True,
    ),
    "ministral3_3b_base": ModelSpec(
        key="ministral3_3b_base",
        hf_id="mistralai/Ministral-3-3B-Base-2512",
        family="ministral3",
        chat_model=False,
    ),
}


def get_model_spec(model_key_or_id: str) -> ModelSpec:
    if model_key_or_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_key_or_id]
    return ModelSpec(
        key=model_key_or_id,
        hf_id=model_key_or_id,
        family="custom",
        chat_model=True,
    )


def resolve_model_config(config: dict[str, Any]) -> tuple[ModelSpec, dict[str, Any]]:
    model_cfg = dict(config.get("model", {}))
    model_key = model_cfg.get("key") or model_cfg.get("hf_id")
    if not model_key:
        raise ValueError("Model config must set either 'key' or 'hf_id'.")
    spec = get_model_spec(model_key)
    model_cfg.setdefault("hf_id", spec.hf_id)
    model_cfg.setdefault("chat_model", spec.chat_model)
    model_cfg.setdefault("trust_remote_code", spec.trust_remote_code)
    model_cfg.setdefault("auto_model_class", spec.auto_model_class)
    model_cfg.setdefault("prefer_processor", spec.prefer_processor)
    return spec, model_cfg


def _hf_from_pretrained_kwargs(model_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "cache_dir": model_cfg.get("cache_dir"),
        "token": os.environ.get("HF_TOKEN") or None,
        "trust_remote_code": model_cfg.get("trust_remote_code", True),
        "local_files_only": bool(model_cfg.get("local_files_only", False)),
    }


def load_text_preprocessor(model_cfg: dict[str, Any]) -> TextPreprocessor:
    from transformers import AutoProcessor, AutoTokenizer

    hf_kwargs = _hf_from_pretrained_kwargs(model_cfg)
    processor = None
    if model_cfg.get("prefer_processor", False):
        try:
            processor = AutoProcessor.from_pretrained(model_cfg["hf_id"], **hf_kwargs)
        except Exception as exc:
            print(
                f"Continuing without AutoProcessor for {model_cfg['hf_id']} "
                f"after processor load failed: {exc}"
            )

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["hf_id"],
            use_fast=True,
            **hf_kwargs,
        )
    except Exception as exc:
        print(
            f"Falling back to use_fast=False for tokenizer {model_cfg['hf_id']} "
            f"after fast-tokenizer load failed: {exc}"
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_cfg["hf_id"],
                use_fast=False,
                **hf_kwargs,
            )
        except Exception:
            if processor is None or getattr(processor, "tokenizer", None) is None:
                raise
            tokenizer = processor.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    return TextPreprocessor(tokenizer=tokenizer, processor=processor)


def load_pretrained_model(
    *,
    model_cfg: dict[str, Any],
    auto_model_class: str,
    torch_dtype: Any,
    device_map: str | None = None,
) -> Any:
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    hf_kwargs = _hf_from_pretrained_kwargs(model_cfg)
    hf_kwargs["torch_dtype"] = torch_dtype
    if device_map is not None:
        hf_kwargs["device_map"] = device_map

    if auto_model_class == "image_text_to_text":
        model_loader = AutoModelForImageTextToText
    else:
        model_loader = AutoModelForCausalLM

    try:
        return model_loader.from_pretrained(model_cfg["hf_id"], **hf_kwargs)
    except ValueError as exc:
        message = str(exc)
        if "does not recognize this architecture" in message or "Unrecognized configuration class" in message:
            raise ValueError(
                f"{message}\n"
                "This environment likely needs a newer Transformers build for the requested model."
            ) from exc
        raise
