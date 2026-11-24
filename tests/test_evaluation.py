import json
import os
import sys
import time

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from coderl import evalplus, evaluation
from coderl.data import load_split
from coderl.forgetting import mean_token_loss
from coderl.stats import mcnemar_exact, paired_bootstrap, wilson_ci


@pytest.fixture(scope="module")
def tiny(tiny_model_dir):
    return AutoModelForCausalLM.from_pretrained(tiny_model_dir), AutoTokenizer.from_pretrained(tiny_model_dir)


def test_generation_ignores_the_models_sampling_defaults(tiny, monkeypatch):
    model, tokenizer = tiny
    monkeypatch.setattr(evaluation, "MAX_NEW_TOKENS", 12)
    config = evaluation.generation_config(tokenizer)
    prompts = ["<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"] * 2
    first = evaluation.generate(model, tokenizer, prompts, config)
    second = evaluation.generate(model, tokenizer, prompts, config)
    assert first == second and first[0] == first[1]  # greedy, despite generation_config.json asking for sampling
    assert model.generation_config.do_sample  # the model's own config is restored afterwards


def test_generation_config_reaches_a_peft_model(tiny):
    """With a PeftModel, `model.generation_config = ...` would stay on the wrapper; generate() installs it inside."""
    from peft import LoraConfig, get_peft_model

    model, tokenizer = tiny
    peft_model = get_peft_model(model, LoraConfig(r=2, task_type="CAUSAL_LM"))
    seen = []
    original = peft_model.get_base_model()._prepare_generation_config

    def spy(generation_config, *args, **kwargs):
        config, model_kwargs = original(generation_config, *args, **kwargs)
        seen.append((config.do_sample, config.repetition_penalty))
        return config, model_kwargs

    peft_model.get_base_model()._prepare_generation_config = spy
    config = evaluation.generation_config(tokenizer)
    config.max_new_tokens = 4
    evaluation.generate(peft_model, tokenizer, ["<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"], config)
    assert seen and seen[-1] == (False, 1.0)
    peft_model.unload()


def test_evaluate_kodcode_end_to_end(tiny, data_dir, monkeypatch):
    model, tokenizer = tiny
    monkeypatch.setattr(evaluation, "MAX_NEW_TOKENS", 16)
    records = load_split(data_dir, "test")
    summary, rows = evaluation.evaluate_kodcode(model, tokenizer, records)
    assert summary["n"] == 2 and summary["pass@1"] == 0.0  # a random model solves nothing
    assert {"passed", "noop", "ruff", "cc", "pylint", "mi", "quality"} <= set(rows[0])
    summary, _ = evaluation.evaluate_kodcode(model, tokenizer, records, n=3)  # best-of-n path
    assert summary["n"] == 2


def test_evalplus_prompt_matches_evalplus(tiny):
    _, tokenizer = tiny
    prompt = evalplus.make_raw_chat_prompt("def f():\n    pass\n", tokenizer)
    assert prompt.startswith("<|im_start|>user\n" + evalplus.INSTRUCTION_PREFIX + "\n```\ndef f():\n    pass\n```")
    assert prompt.endswith(evalplus.RESPONSE_PREFIX + "\n```python\n")


def test_forgetting_loss(tiny):
    model, tokenizer = tiny
    model.train()
    loss = mean_token_loss(model, tokenizer, ["def f():\n    return 1\n" * 5, "short text"], batch_size=2)
    assert loss > 0 and model.training  # mode restored
    assert loss == pytest.approx(mean_token_loss(model, tokenizer, ["def f():\n    return 1\n" * 5, "short text"]))


def test_stats():
    lo, hi = wilson_ci(148, 164)
    assert round(100 * lo, 1) == 84.7 and round(100 * hi, 1) == 93.9
    assert round(mcnemar_exact([True] * 11 + [False] * 14, [False] * 11 + [True] * 14), 3) == 0.690
    diff, lo, hi = paired_bootstrap([1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0])
    assert diff == 0.5 and lo <= diff <= hi


def test_scoring_timeout_kills_the_whole_process_group(tmp_path):
    """EvalPlus can hang after a worker dies; the timeout must also kill uvx's child and its workers."""
    marker = tmp_path / "grandchild_survived"
    grandchild = f"import time; time.sleep(3); open({str(marker)!r}, 'w')"
    child = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {grandchild!r}]); time.sleep(60)"
    start = time.time()
    with pytest.raises(evalplus.ScoringError, match="timed out"):
        evalplus._run([sys.executable, "-c", child], dict(os.environ), timeout=1)
    assert time.time() - start < 10
    time.sleep(4)
    assert not marker.exists()
    with pytest.raises(evalplus.ScoringError, match="exited with 3"):
        evalplus._run([sys.executable, "-c", "raise SystemExit(3)"], dict(os.environ), timeout=10)


def test_scoring_retries_with_one_worker(tmp_path, monkeypatch):
    workers = []

    def fake_run(cmd, env, timeout, label):
        if "evalplus.sanitize" in cmd:
            (tmp_path / "humaneval-sanitized.jsonl").write_text(json.dumps({"task_id": "HumanEval/0", "solution": "x"}))
            return
        workers.append(cmd[-1])
        if cmd[-1] != "1":
            raise evalplus.ScoringError("BrokenProcessPool")
        result = {"eval": {"HumanEval/0": [{"base_status": "pass", "plus_status": "fail"}]}}
        (tmp_path / "humaneval-sanitized_eval_results.json").write_text(json.dumps(result))

    monkeypatch.setattr(evalplus, "_run", fake_run)
    monkeypatch.setattr(evalplus.shutil, "which", lambda _: "/usr/bin/uvx")
    scored = evalplus.score({"HumanEval/0": "x"}, "humaneval", tmp_path)
    assert workers == ["2", "1"]
    assert scored == {"HumanEval/0": {"base": True, "plus": False, "sanitized": "x"}}


def test_run_keeps_kodcode_results_when_evalplus_fails(tiny, data_dir, monkeypatch):
    import run

    def broken(*args, **kwargs):
        raise evalplus.ScoringError("timed out")

    monkeypatch.setattr(evaluation, "MAX_NEW_TOKENS", 16)
    monkeypatch.setattr(run, "evaluate_evalplus", broken)
    model, tokenizer = tiny
    out = run.evaluate(model, tokenizer, {"eval_limit": None, "evalplus": True}, data_dir, data_dir / "runs")
    assert out["kodcode_test"]["n"] == 2 and out["humaneval"] == out["mbpp"] == {"error": "timed out"}


@pytest.mark.network
@pytest.mark.parametrize("name, expected", [("humaneval", 164), ("mbpp", 376)])
def test_evalplus_scores_canonical_solutions(tmp_path, name, expected):
    """HumanEval+: all 164. MBPP+: all but Mbpp/630 (the sanitizer drops its helper) and Mbpp/255 when the machine
    is slow (EvalPlus' 60 s cap). MBPP+ needs ~8 GB of free RAM (~2 GB main process, up to ~4 GB per worker)."""
    tasks = evalplus.load_tasks(name)
    scored = evalplus.score({t["task_id"]: t["prompt"] + t["canonical_solution"] for t in tasks}, name, tmp_path)
    assert sum(r["plus"] for r in scored.values()) >= expected
