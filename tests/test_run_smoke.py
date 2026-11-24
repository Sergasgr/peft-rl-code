"""End to end on CPU: every adapter config of the study trains for 2 GRPO steps on a tiny model and gets evaluated."""

import json
import shutil
from pathlib import Path

import pytest

import run

ROOT = Path(__file__).resolve().parent.parent
TINY = {
    "dtype": "float32",
    "attn_implementation": "eager",
    "max_steps": 2,
    "prompts_per_step": 2,
    "num_generations": 2,
    "micro_batch_size": 2,
    "max_completion_length": 16,
    "warmup_steps": 0,
    "gradient_checkpointing": False,
    "logging_steps": 1,
    "evalplus": False,
    "forgetting": False,
    "eval_limit": 2,
}


def _experiment(tmp_path, tiny_model_dir, adapter, **params) -> Path:
    exp = tmp_path / "experiments" / (adapter or "full") / "smoke"
    exp.mkdir(parents=True)
    if adapter:
        shutil.copyfile(ROOT / "configs" / "adapters" / f"{adapter}.json", exp / "adapter_config.json")
    (exp / "training_params.json").write_text(json.dumps({**TINY, "model_id": str(tiny_model_dir), **params}))
    return exp


@pytest.mark.slow
@pytest.mark.parametrize("adapter", ["lora", "dora", "pissa", "oft", "vera", "ia3", "tinylora", None])
def test_every_method_trains_and_evaluates(tmp_path, monkeypatch, tiny_model_dir, data_dir, adapter):
    monkeypatch.setenv("CODERL_OUTPUT", str(tmp_path))
    monkeypatch.setattr(run, "evaluate_kodcode", _short(run.evaluate_kodcode))
    exp = _experiment(tmp_path, tiny_model_dir, adapter, learning_rate=1e-3)
    result = json.loads(run.main([str(exp), "--data-dir", str(data_dir)]).read_text())
    assert result["run_info"]["status"] == "success"
    assert result["eval"]["kodcode_test"]["n"] == 2 and len(result["eval"]["kodcode_test"]["per_task"]) == 2
    info = result["train_info"]
    assert 0 < info["num_trainable_params"] <= info["num_total_params"]
    if adapter:
        assert info["num_trainable_params"] < info["num_total_params"]
        assert (tmp_path / "checkpoints" / f"{adapter}--smoke" / "adapter_config.json").exists()
    assert any("monitor/pass_rate" in entry for entry in info["log_history"])


@pytest.mark.slow
@pytest.mark.parametrize("reward", ["tests+quality", "quality"])
def test_reward_modes_train(tmp_path, monkeypatch, tiny_model_dir, data_dir, reward):
    monkeypatch.setenv("CODERL_OUTPUT", str(tmp_path))
    monkeypatch.setattr(run, "evaluate_kodcode", _short(run.evaluate_kodcode))
    exp = _experiment(tmp_path, tiny_model_dir, "lora", reward=reward)
    result = json.loads(run.main([str(exp), "--data-dir", str(data_dir)]).read_text())
    names = {k for entry in result["train_info"]["log_history"] for k in entry}
    expected = "rewards/gated_quality_reward/mean" if reward == "tests+quality" else "rewards/quality_reward/mean"
    assert expected in names


def _short(evaluate_kodcode):
    """Evaluation with 16 new tokens instead of 768, so the smoke test stays fast."""
    from coderl import evaluation

    def wrapper(*args, **kwargs):
        saved = evaluation.MAX_NEW_TOKENS
        evaluation.MAX_NEW_TOKENS = 16
        try:
            return evaluate_kodcode(*args, **kwargs)
        finally:
            evaluation.MAX_NEW_TOKENS = saved

    return wrapper


def _varied_reward(completions, **kwargs):
    """A random model fails every test, so all advantages are 0 and nothing trains. This reward varies."""
    return [float(len(c[0]["content"]) % 3) for c in completions]


@pytest.mark.slow
@pytest.mark.parametrize("adapter", ["lora", "dora", "pissa", "oft", "vera", "ia3", "tinylora", None])
def test_every_method_gets_gradients(tmp_path, monkeypatch, tiny_model_dir, data_dir, adapter):
    monkeypatch.setenv("CODERL_OUTPUT", str(tmp_path))
    monkeypatch.setattr(run, "evaluate_kodcode", _short(run.evaluate_kodcode))
    monkeypatch.setattr(run.rewards, "build", lambda mode, weight: ([_varied_reward], [1.0]))
    # all 16-token completions are truncated, and truncated ones are masked by default: unmask them here
    exp = _experiment(
        tmp_path,
        tiny_model_dir,
        adapter,
        num_generations=4,
        prompts_per_step=2,
        micro_batch_size=4,
        gradient_checkpointing=True,
        learning_rate=1e-3,
        mask_truncated_completions=False,
    )
    result = json.loads(run.main([str(exp), "--data-dir", str(data_dir)]).read_text())
    grad_norms = [float(e["grad_norm"]) for e in result["train_info"]["log_history"] if "grad_norm" in e]
    assert grad_norms and max(grad_norms) > 0


@pytest.mark.slow
def test_baseline_with_best_of_n(tmp_path, monkeypatch, tiny_model_dir, data_dir):
    import baseline

    monkeypatch.setenv("CODERL_OUTPUT", str(tmp_path))
    monkeypatch.setattr(baseline, "evaluate_kodcode", _short(baseline.evaluate_kodcode))
    monkeypatch.setattr(run, "evaluate_kodcode", _short(run.evaluate_kodcode))
    path = baseline.main(
        [
            "--model-id",
            str(tiny_model_dir),
            "--data-dir",
            str(data_dir),
            "--best-of-n",
            "3",
            "--eval-limit",
            "2",
            "--no-evalplus",
        ]
    )
    result = json.loads(path.read_text())
    assert set(result["eval"]) == {"kodcode_valid", "kodcode_test", "kodcode_test_best_of_3"}
    saved = tmp_path / "runs" / f"baseline--{tiny_model_dir.name}"
    for split in ("kodcode_valid", "kodcode_test", "kodcode_test_best_of_3"):
        rows = [json.loads(line) for line in (saved / f"{split}.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and "code" in rows[0] and "passed" in rows[0]
