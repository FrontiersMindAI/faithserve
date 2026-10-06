"""Offline fixtures: a tiny tokenizer trained in memory, with ChatML-style chat templates."""

from __future__ import annotations

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

from faithserve.reference import load_tokenizer

_TURNS = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
_TOOLS = (
    "{% if tools %}<|im_start|>system\n# Tools\n"
    "{% for t in tools %}{{ t | tojson }}\n{% endfor %}<|im_end|>\n{% endif %}"
)
TEMPLATES = {
    "default": "{{ bos_token }}" + _TOOLS + _TURNS,  # BOS in the template, renders tools
    "no_tools": "{{ bos_token }}" + _TURNS,
    "no_bos": _TOOLS + _TURNS,  # no BOS in the template; the tokenizer adds one itself
}
_CORPUS = [
    "What is the capital of France? You are a terse assistant. Answer in one sentence.",
    "system user assistant tools function name description parameters properties required",
    "alpha bravo charlie delta echo foxtrot golf hotel india juliett kilo lima mike november",
]


def _train() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=["<s>", "</s>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(_CORPUS, trainer)
    # Like Llama-style tokenizers: encoding with special tokens prepends BOS.
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", tok.token_to_id("<s>"))]
    )
    return tok


@pytest.fixture(scope="session")
def tokenizer_dirs(tmp_path_factory):
    """One saved tokenizer directory per chat-template variant."""
    dirs = {}
    for name, template in TEMPLATES.items():
        fast = PreTrainedTokenizerFast(tokenizer_object=_train(), bos_token="<s>", eos_token="</s>")
        fast.chat_template = template
        path = tmp_path_factory.mktemp(f"tok-{name}")
        fast.save_pretrained(str(path))
        dirs[name] = str(path)
    return dirs


@pytest.fixture(scope="session")
def tokenizer(tokenizer_dirs):
    return load_tokenizer(tokenizer_dirs["default"])
