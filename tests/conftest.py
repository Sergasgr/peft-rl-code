"""Shared fixtures: KodCode-like problems and a tiny random Qwen2 model with a ChatML tokenizer (no downloads)."""

import json
from pathlib import Path

import pytest


def _task(qid, question, solution, test, name, decl):
    return {
        "question_id": qid,
        "question": question,
        "solution": solution,
        "test": test,
        "test_info": [{"function_name": name, "function_declaration": decl, "parameter_list": "", "docstring": None}],
    }


TASKS = [
    _task(
        "add",
        "Return the sum of two integers.",
        "def add(a, b):\n    return a + b\n",
        "def test_add():\n    assert add(1, 2) == 3\n    assert add(-1, 1) == 0\n",
        "add",
        "def add(a, b):",
    ),
    _task(
        "even",
        "Return True if n is even.",
        "def is_even(n):\n    return n % 2 == 0\n",
        "def test_even():\n    assert is_even(2)\n    assert not is_even(3)\n",
        "is_even",
        "def is_even(n):",
    ),
    _task(
        "rev",
        "Reverse a string.",
        "def reverse(s):\n    return s[::-1]\n",
        "def test_rev():\n    assert reverse('abc') == 'cba'\n",
        "reverse",
        "def reverse(s):",
    ),
    _task(
        "max",
        "Return the largest number of a list.",
        "def largest(xs):\n    return max(xs)\n",
        "import pytest\n\ndef test_max():\n    assert largest([1, 5, 2]) == 5\n    with pytest.raises(ValueError):\n"
        "        largest([])\n",
        "largest",
        "def largest(xs):",
    ),
    _task(
        "vowels",
        "Count the vowels in a string.",
        "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
        "def test_vowels():\n    assert count_vowels('Hello') == 2\n",
        "count_vowels",
        "def count_vowels(s):",
    ),
    _task(
        "sign",
        "Return -1, 0 or 1 depending on the sign of n.",
        "def sign(n):\n    if n > 0:\n        return 1\n    if n < 0:\n        return -1\n    return 0\n",
        "def test_sign():\n    assert sign(5) == 1\n    assert sign(-2) == -1\n    assert sign(0) == 0\n",
        "sign",
        "def sign(n):",
    ),
    _task(
        "fact",
        "Return n factorial.",
        "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n",
        "def test_fact():\n    assert factorial(5) == 120\n    assert factorial(0) == 1\n",
        "factorial",
        "def factorial(n):",
    ),
    # filtered out: blocked library, wrong reference solution, no function info
    _task(
        "pandas",
        "Load a CSV.",
        "import pandas\ndef load(p):\n    return pandas.read_csv(p)\n",
        "def test_load():\n    pass\n",
        "load",
        "def load(p):",
    ),
    _task(
        "broken",
        "Double a number.",
        "def double(x):\n    return x + 1\n",
        "def test_double():\n    assert double(3) == 6\n",
        "double",
        "def double(x):",
    ),
    {
        "question_id": "class",
        "question": "A stack class.",
        "solution": "class Stack:\n    pass\n",
        "test": "def test_stack():\n    assert Stack()\n",
        "test_info": None,
    },
]

CHATML = (
    "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


@pytest.fixture(scope="session")
def kodcode_jsonl(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("source") / "kodcode.jsonl"
    path.write_text("".join(json.dumps(t) + "\n" for t in TASKS), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory, kodcode_jsonl) -> Path:
    from coderl.data import prepare

    out = tmp_path_factory.mktemp("data") / "kodcode"
    prepare(out, source=str(kodcode_jsonl), n_valid=2, n_test=2, seed=0)
    return out


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> Path:
    """A 2-layer Qwen2 with random weights and a byte-level BPE tokenizer trained on this repo's code."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GenerationConfig, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    out = tmp_path_factory.mktemp("tiny-qwen2")
    corpus = [p.read_text(encoding="utf-8") for p in Path(__file__).resolve().parent.parent.glob("**/*.py")]
    corpus += [json.dumps(t) for t in TASKS]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        corpus,
        trainers.BpeTrainer(
            vocab_size=800,
            special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tokenizer.chat_template = CHATML
    tokenizer.save_pretrained(out)

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = Qwen2ForCausalLM(config)
    # what Qwen ships: sampling with repetition_penalty 1.1, which evaluation must override
    model.generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
    )
    model.save_pretrained(out)
    return out
