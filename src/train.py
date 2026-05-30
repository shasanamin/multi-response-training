from __future__ import annotations

import math
from inspect import signature
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from formatting import build_training_texts
from jsonl_io import iter_jsonl
from models import ModelSpec, load_pretrained_model, load_text_preprocessor


class SupervisedFineTuningDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        spec: ModelSpec,
        max_length: int,
    ) -> None:
        self.examples: list[dict[str, list[int]]] = []
        for record in records:
            prompt_text, full_text = build_training_texts(
                tokenizer,
                record["prompt"],
                record["response"],
                spec,
            )
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
            )["input_ids"]
            tokenized = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            input_ids = list(tokenized["input_ids"])
            attention_mask = list(tokenized["attention_mask"])
            labels = list(input_ids)
            prompt_length = min(len(prompt_ids), len(labels))
            labels[:prompt_length] = [-100] * prompt_length
            if all(label == -100 for label in labels):
                continue
            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class SupervisedCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}

        for feature in features:
            pad_width = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad_width)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_width)
            batch["labels"].append(feature["labels"] + [-100] * pad_width)

        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in batch.items()
        }


def load_records(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def _default_target_modules(model: Any, model_family: str) -> list[str]:
    family_preferences = {
        "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        "gemma3": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        "gemma4": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        "ministral3": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        "qwen35": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        "custom": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    }
    fallback_preferences = ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "qkv_proj", "up_proj", "down_proj", "gate_proj"]
    available_suffixes = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules() if name}

    preferred = family_preferences.get(model_family, fallback_preferences)
    if model_family == "gemma4":
        exact_linear_matches = [
            name
            for name, module in model.named_modules()
            if name
            and isinstance(module, torch.nn.Linear)
            and name.rsplit(".", 1)[-1] in preferred
        ]
        if exact_linear_matches:
            return exact_linear_matches

    selected = [name for name in preferred if name in available_suffixes]
    if selected:
        return selected

    fallback = [name for name in fallback_preferences if name in available_suffixes]
    if fallback:
        return fallback

    preview = ", ".join(sorted(available_suffixes)[:25])
    raise ValueError(
        f"Could not infer LoRA target modules for model family '{model_family}'. "
        f"Available module suffixes include: {preview}"
    )


def _prepare_text_preprocessor(model_cfg: dict[str, Any]) -> Any:
    return load_text_preprocessor(model_cfg)


def _prepare_model(model_cfg: dict[str, Any], training_cfg: dict[str, Any]) -> Any:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = load_pretrained_model(
        model_cfg=model_cfg,
        auto_model_class=str(model_cfg.get("auto_model_class", "causal_lm")),
        torch_dtype=dtype,
    )
    if training_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model


def _maybe_apply_lora(model: Any, spec: ModelSpec, training_cfg: dict[str, Any]) -> Any:
    if not training_cfg.get("use_lora", True):
        return model

    from peft import LoraConfig, get_peft_model

    target_modules = training_cfg.get("target_modules") or _default_target_modules(model, spec.family)
    lora_config = LoraConfig(
        r=int(training_cfg.get("lora_r", 16)),
        lora_alpha=int(training_cfg.get("lora_alpha", 32)),
        lora_dropout=float(training_cfg.get("lora_dropout", 0.05)),
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model.enable_input_require_grads()
    return get_peft_model(model, lora_config)


def train_model(
    *,
    train_path: str | Path,
    eval_path: str | Path,
    output_dir: str | Path,
    spec: ModelSpec,
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    seed: int,
) -> tuple[Path, dict[str, float]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_records = load_records(train_path)
    eval_records = load_records(eval_path)

    tokenizer = _prepare_text_preprocessor(model_cfg)
    train_dataset = SupervisedFineTuningDataset(
        train_records,
        tokenizer,
        spec,
        max_length=int(training_cfg.get("max_length", 1024)),
    )
    eval_dataset = SupervisedFineTuningDataset(
        eval_records,
        tokenizer,
        spec,
        max_length=int(training_cfg.get("max_length", 1024)),
    )

    model = _prepare_model(model_cfg, training_cfg)
    model = _maybe_apply_lora(model, spec, training_cfg)

    save_steps = int(training_cfg.get("save_steps", 25))
    eval_steps = int(training_cfg.get("eval_steps", 25))
    eval_enabled = len(eval_dataset) > 0
    load_best_model_at_end = bool(
        eval_enabled
        and eval_steps > 0
        and save_steps > 0
        and save_steps % eval_steps == 0
    )
    if eval_enabled and not load_best_model_at_end:
        print(
            "Disabling load_best_model_at_end because save_steps "
            f"({save_steps}) is not a round multiple of eval_steps ({eval_steps})."
        )

    training_kwargs = dict(
        output_dir=str(output_path),
        overwrite_output_dir=False,
        seed=seed,
        data_seed=seed,
        learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(training_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 8)),
        num_train_epochs=float(training_cfg.get("num_train_epochs", 1.0)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        logging_steps=int(training_cfg.get("logging_steps", 5)),
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=int(training_cfg.get("save_total_limit", 2)),
        load_best_model_at_end=load_best_model_at_end,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        logging_dir=str(output_path / "tb"),
        save_safetensors=True,
        optim=training_cfg.get("optim", "adamw_torch"),
    )
    if load_best_model_at_end:
        training_kwargs["metric_for_best_model"] = "eval_loss"
        training_kwargs["greater_is_better"] = False

    strategy_name = "steps" if eval_enabled else "no"
    init_signature = signature(TrainingArguments.__init__)
    if "evaluation_strategy" in init_signature.parameters:
        training_kwargs["evaluation_strategy"] = strategy_name
    else:
        training_kwargs["eval_strategy"] = strategy_name

    supported_kwargs = {
        key: value
        for key, value in training_kwargs.items()
        if key in init_signature.parameters
    }
    dropped_kwargs = sorted(set(training_kwargs) - set(supported_kwargs))
    if dropped_kwargs:
        print(
            "Skipping unsupported TrainingArguments kwargs for this Transformers build: "
            + ", ".join(dropped_kwargs)
        )

    args = TrainingArguments(**supported_kwargs)

    max_steps = training_cfg.get("max_steps")
    if max_steps is not None:
        args.max_steps = int(max_steps)
        args.num_train_epochs = math.ceil(int(max_steps) / max(len(train_dataset), 1))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        data_collator=SupervisedCollator(tokenizer),
    )

    resume_from_checkpoint = training_cfg.get("resume_from_checkpoint")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = trainer.evaluate() if len(eval_dataset) > 0 else {}
    trainer.save_model()
    tokenizer.save_pretrained(output_path)
    return output_path, {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
