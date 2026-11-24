import json

import pytest

from coderl import rewards
from coderl.data import extract_code, load_split


def test_extract_code():
    assert extract_code("Here:\n```python\ndef f():\n    return 1\n```\nDone.") == "def f():\n    return 1"
    assert extract_code("```\nx = 1\n```") == "x = 1"
    assert extract_code("```python\ndef f():\n    return") == "def f():\n    return"  # cut before closing fence
    assert extract_code("def f():\n    return 1") == "def f():\n    return 1"  # no fence


def test_prepare_filters_and_splits(data_dir):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    assert manifest["filters"] == {
        "source_rows": 10,
        "function_tasks_without_blocked_libraries": 8,
        "reference_solution_passes": 7,
    }
    splits = {name: load_split(data_dir, name) for name in ("train", "valid", "test")}
    assert [len(s) for s in splits.values()] == [3, 2, 2]
    ids = [r["id"] for s in splits.values() for r in s]
    assert len(set(ids)) == 7 and "broken" not in ids and "pandas" not in ids
    record = splits["train"][0]
    assert record["tests"].startswith("from solution import *\n")
    assert record["prompt"][0]["role"] == "user" and "function declaration is def" in record["prompt"][0]["content"]
    assert set(record) == {"id", "prompt", "tests", "entry_point", "ref_cc", "ref_ruff"}


def test_prepare_drops_long_prompts(kodcode_jsonl, tiny_model_dir, tmp_path):
    from transformers import AutoTokenizer

    from coderl.data import prepare, prompt_tokens

    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
    lengths = sorted(
        prompt_tokens(tokenizer, r["prompt"]) for r in load_split(_prepared(kodcode_jsonl, tmp_path), "train")
    )
    with pytest.raises(SystemExit):  # too few problems left once every prompt is over the limit
        prepare(
            tmp_path / "short",
            str(kodcode_jsonl),
            n_valid=2,
            n_test=2,
            tokenizer=tokenizer,
            max_prompt_tokens=lengths[0] - 1,
        )


def _prepared(source, tmp_path):
    from coderl.data import prepare

    prepare(tmp_path / "all", str(source), n_valid=2, n_test=2)
    return tmp_path / "all"


def test_load_split_rejects_changed_data(data_dir, tmp_path):
    copy = tmp_path / "kodcode"
    copy.mkdir()
    for f in data_dir.iterdir():
        (copy / f.name).write_text(f.read_text())
    (copy / "test.jsonl").write_text((copy / "test.jsonl").read_text() + "\n{}\n")
    with pytest.raises(SystemExit):
        load_split(copy, "test")


TESTS = "from solution import *\n\ndef test_add():\n    assert add(2, 3) == 5\n"
GOOD = [{"role": "assistant", "content": "```python\ndef add(a, b):\n    return a + b\n```"}]
LINTY = [{"role": "assistant", "content": "```python\nimport os\n\ndef add(a, b):\n    return a + b\n```"}]
WRONG = [{"role": "assistant", "content": "```python\ndef add(a, b):\n    return a - b\n```"}]
NOOP = [{"role": "assistant", "content": "```python\ndef add(a, b):\n    return 0\n```"}]


def _kwargs(n):
    return {"tests": [TESTS] * n, "entry_point": ["add"] * n, "ref_cc": [1] * n}


def test_reward_modes():
    completions = [GOOD, LINTY, WRONG]
    assert rewards.tests_reward(completions, **_kwargs(3)) == [1.0, 1.0, 0.0]
    assert rewards.gated_quality_reward(completions, **_kwargs(3)) == [1.0, 0.75, 0.0]  # wrong code gets nothing
    assert rewards.quality_reward(completions, **_kwargs(3)) == [1.0, 0.75, 1.0]  # ... unless nothing checks it
    funcs, weights = rewards.build("tests+quality", quality_weight=0.5)
    assert [f.__name__ for f in funcs] == ["tests_reward", "gated_quality_reward", "monitor"]
    assert weights == [1.0, 0.5, 0.0]
    with pytest.raises(ValueError):
        rewards.build("lint", 0.5)


def test_monitor_logs_without_rewarding():
    logged = {}
    out = rewards.monitor([GOOD, WRONG], log_metric=logged.__setitem__, **_kwargs(2))
    assert out == [0.0, 0.0]
    assert logged["monitor/pass_rate"] == 0.5 and logged["monitor/noop_rate"] == 0.0
    assert logged["monitor/no_code_block"] == 0.0
    rewards.monitor([GOOD, NOOP], log_metric=logged.__setitem__, **_kwargs(2))
    assert logged["monitor/noop_rate"] == 0.5
