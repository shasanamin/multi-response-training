from __future__ import annotations

from typing import Any

from models import ModelSpec


def _has_chat_template(tokenizer: Any) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def build_generation_prompt(tokenizer: Any, prompt: str, spec: ModelSpec) -> str:
    if spec.chat_model and _has_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        "Below is an instruction. Provide a direct and helpful answer.\n\n"
        f"### Instruction:\n{prompt.strip()}\n\n"
        "### Response:\n"
    )


def build_training_texts(tokenizer: Any, prompt: str, response: str, spec: ModelSpec) -> tuple[str, str]:
    if spec.chat_model and _has_chat_template(tokenizer):
        prompt_text = build_generation_prompt(tokenizer, prompt, spec)
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        return prompt_text, full_text

    prompt_text = build_generation_prompt(tokenizer, prompt, spec)
    eos = tokenizer.eos_token or ""
    full_text = f"{prompt_text}{response}{eos}"
    return prompt_text, full_text
