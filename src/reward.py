from __future__ import annotations

from typing import Any

import torch


class RewardModelScorer:
    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: str | None = None,
        token: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            token=token,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()

    def _format_pair(self, prompt: str, response: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            formatted = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            bos_token = getattr(self.tokenizer, "bos_token", None)
            if bos_token and formatted.startswith(bos_token):
                formatted = formatted[len(bos_token) :]
            return formatted
        return f"User: {prompt}\nAssistant: {response}"

    def score(self, prompts: list[str], responses: list[str], batch_size: int = 8) -> list[float]:
        scores: list[float] = []
        device = next(self.model.parameters()).device

        for start in range(0, len(prompts), batch_size):
            prompt_batch = prompts[start : start + batch_size]
            response_batch = responses[start : start + batch_size]
            texts = [self._format_pair(p, r) for p, r in zip(prompt_batch, response_batch)]
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits
            if logits.ndim == 2 and logits.shape[-1] > 1:
                batch_scores = logits[:, -1]
            else:
                batch_scores = logits.view(-1)
            scores.extend(batch_scores.detach().float().cpu().tolist())
        return scores
