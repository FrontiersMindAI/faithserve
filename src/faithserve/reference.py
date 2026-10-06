"""The reference side: the model's own tokenizer and chat template.

Only tokenizer files are ever loaded, never model weights.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class TokenizerError(Exception):
    """The reference tokenizer could not be loaded or used."""


@dataclass
class Rendered:
    text: str
    tokens: int  # the chat-template convention: no extra special tokens added
    tokens_with_special: int  # same text, tokenizer allowed to add its own special tokens
    starts_with_bos: bool


def load_tokenizer(model_id: str):
    """Load the tokenizer (and chat template) for a Hugging Face model id or local path."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:  # transformers raises many types here; all mean "no tokenizer"
        raise TokenizerError(_explain_load_failure(model_id, exc)) from None
    if not getattr(tokenizer, "chat_template", None):
        raise TokenizerError(
            f"{model_id} has no chat template, so there is no reference prompt format to compare against."
        )
    return tokenizer


def _explain_load_failure(model_id: str, exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "gated" in text or "401" in text or "403" in text:
        return (
            f"{model_id} is gated or private. Accept its licence on huggingface.co and "
            "authenticate (set HF_TOKEN or run `hf auth login`), then retry."
        )
    if "not a valid model identifier" in text or "404" in text or "not found" in text:
        return (
            f"Could not find {model_id!r} on the Hugging Face Hub. Check the id; if the repo is private or gated, "
            "set HF_TOKEN."
        )
    if "connect" in text or "offline" in text or "timed out" in text:
        return f"Could not download the tokenizer for {model_id} (network problem or offline mode)."
    first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return f"Could not load the tokenizer for {model_id}: {first_line}"


def render(tokenizer, messages: list[dict], tools: list[dict] | None = None) -> Rendered:
    """Render a conversation the way the model's template says, ready for generation."""
    try:
        text = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
    except Exception as exc:  # templates raise on shapes they reject, e.g. no system role
        raise TokenizerError(str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__) from None
    plain = tokenizer(text, add_special_tokens=False)["input_ids"]
    with_special = tokenizer(text, add_special_tokens=True)["input_ids"]
    bos = tokenizer.bos_token
    return Rendered(
        text=text,
        tokens=len(plain),
        tokens_with_special=len(with_special),
        starts_with_bos=bool(bos) and text.startswith(bos),
    )


def supports_tools(tokenizer, messages: list[dict], tools: list[dict]) -> bool:
    """A template supports tools if passing them changes the prompt."""
    try:
        return render(tokenizer, messages, tools).text != render(tokenizer, messages).text
    except TokenizerError:
        return False


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])
