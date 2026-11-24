import json
import subprocess
from pathlib import Path

import pytest

import compare
import make_experiments
import table
from coderl.config import DEFAULTS, experiment_name, git_state, load_params, result_path

ROOT = Path(__file__).resolve().parent.parent


def test_params_override_and_typos(tmp_path):
    exp = tmp_path / "experiments" / "lora" / "x"
    exp.mkdir(parents=True)
    (exp / "training_params.json").write_text(json.dumps({"learning_rate": 3e-5}))
    params = load_params(exp)
    assert params["learning_rate"] == 3e-5
    assert params["max_steps"] == json.loads(DEFAULTS.read_text())["max_steps"]
    (exp / "training_params.json").write_text(json.dumps({"learnig_rate": 3e-5}))
    with pytest.raises(SystemExit):
        load_params(exp)


def test_experiment_name_and_result_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("CODERL_OUTPUT", str(tmp_path))
    assert experiment_name(tmp_path / "experiments" / "lora" / "lr1e-5-tests") == "lora/lr1e-5-tests"
    clean = {"branch": "main", "dirty": False}
    assert result_path("lora/a", "success", clean) == tmp_path / "results" / "lora--a.json"
    assert result_path("lora/a", "success", {**clean, "dirty": True}).parent.name == "temporary_results"
    assert result_path("lora/a", "success", {**clean, "branch": "dev"}).parent.name == "temporary_results"
    assert result_path("lora/a", "cancelled", clean).parent.name == "cancelled_results"


def test_git_state_ignores_results_and_queue(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-b", "main")
    (tmp_path / "queue.txt").write_text("a\n")
    (tmp_path / "run.py").write_text("x = 1\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "init")
    (tmp_path / "queue.txt").write_text("b\n")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "new.json").write_text("{}")
    assert git_state(tmp_path) == {"branch": "main", "sha": git_state(tmp_path)["sha"], "dirty": False}
    (tmp_path / "run.py").write_text("x = 2\n")
    assert git_state(tmp_path)["dirty"]


def test_make_experiments_stages(tmp_path):
    (tmp_path / "configs" / "adapters").mkdir(parents=True)
    (tmp_path / "configs" / "adapters" / "lora.json").write_text('{"peft_type": "LORA"}')
    spec = {
        "pilot": {"model_id": "small", "method": "lora", "learning_rate": 1e-5, "rewards": ["tests"]},
        "methods": {
            "lora": {
                "adapter": "configs/adapters/lora.json",
                "grid": [3e-6, 1e-5],
                "selected_lr": 1e-5,
                "extra_seeds": [1],
            },
            "full": {
                "adapter": None,
                "grid": [1e-6],
                "selected_lr": None,
                "extra_seeds": [],
                "training_params": {"optim": "adamw_bnb_8bit"},
            },
        },
    }
    (tmp_path / "methods.json").write_text(json.dumps(spec))
    lines = make_experiments.main(tmp_path)
    assert lines == [
        "experiments/pilot/lora-lr1e-5-tests",
        "experiments/lora/lr3e-6-tests",
        "experiments/lora/lr1e-5-tests",
        "experiments/full/lr1e-6-tests",
        "experiments/lora/lr1e-5-tests+quality",
        "experiments/lora/lr1e-5-quality",
        "experiments/lora/lr1e-5-tests-seed1",
        "experiments/lora/lr1e-5-tests+quality-seed1",
    ]
    assert not (tmp_path / "experiments" / "full" / "lr1e-6-tests" / "adapter_config.json").exists()
    params = json.loads((tmp_path / "experiments" / "full" / "lr1e-6-tests" / "training_params.json").read_text())
    assert params == {"optim": "adamw_bnb_8bit", "learning_rate": 1e-6, "reward": "tests"}
    for line in lines:  # every generated setting exists in the defaults
        load_params(tmp_path / line)


def test_repo_methods_json_is_valid(tmp_path):
    for adapter in (ROOT / "configs" / "adapters").glob("*.json"):
        from peft import PeftConfig

        (tmp_path / adapter.stem).mkdir()
        (tmp_path / adapter.stem / "adapter_config.json").write_text(adapter.read_text())
        assert PeftConfig.from_pretrained(str(tmp_path / adapter.stem)).task_type == "CAUSAL_LM"
    assert make_experiments.lr_tag(4.5e-3) == "4.5e-3" and make_experiments.lr_tag(1e-4) == "1e-4"


def _result(name, lr, valid, reward="tests"):
    return {
        "run_info": {"experiment_name": name, "params": {"reward": reward, "seed": 0, "learning_rate": lr}},
        "eval": {"kodcode_valid": {"pass@1": valid}, "kodcode_test": {"pass@1": 0.5, "passing": {}}},
        "train_info": {"num_trainable_params": 10},
    }


def test_table_selects_lr_on_valid_only(tmp_path):
    (tmp_path / "results").mkdir()
    for i, (lr, valid) in enumerate([(3e-6, 0.40), (1e-5, 0.45), (3e-5, 0.45)]):
        (tmp_path / "results" / f"lora--{i}.json").write_text(json.dumps(_result(f"lora/lr{lr}", lr, valid)))
    table.main(tmp_path)
    text = (tmp_path / "RESULTS.md").read_text()
    assert "| lora | 1e-05 | 45.0 | ✓ |" in text  # tie with 3e-5 goes to the lower rate
    assert "| lora | 3e-05 | 45.0 |  |" in text


def test_table_lists_best_of_n_baseline(tmp_path):
    (tmp_path / "results").mkdir()
    base = _result("baseline/Qwen2.5-Coder-1.5B-Instruct", 1e-5, 0.4)
    base["eval"]["kodcode_test_best_of_8"] = {"pass@1": 0.25, "passing": {"quality": 0.9}}
    (tmp_path / "results" / "baseline.json").write_text(json.dumps(base))
    table.main(tmp_path)
    text = (tmp_path / "RESULTS.md").read_text()
    assert "| baseline/Qwen2.5-Coder-1.5B-Instruct | 50.0 |" in text
    assert "| baseline/Qwen2.5-Coder-1.5B-Instruct, best-of-8 | 25.0 |" in text
    assert "| lora |" not in text  # baselines never enter the learning-rate table


def _per_task(tmp_path, name, bits, split="kodcode_test"):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"eval": {split: {"per_task": bits}}}))
    return path


def test_compare_single_runs_and_seeds(tmp_path):
    a = _per_task(tmp_path, "a", "1" * 30 + "0" * 10)
    b = _per_task(tmp_path, "b", "0" * 30 + "1" * 10)
    out = compare.compare([a], [b])
    assert (out["only_a"], out["only_b"]) == (30, 10) and out["mcnemar_p"] < 0.05 and out["detectable"]
    assert out["pass@1_a"] == 0.75 and out["pass@1_b"] == 0.25

    same = compare.compare([a], [_per_task(tmp_path, "c", "1" * 30 + "0" * 10)])
    assert same["mcnemar_p"] == 1.0 and not same["detectable"]

    seeds = compare.compare([a, _per_task(tmp_path, "a2", "0" * 40)], [b])  # A's per-task rate: 0.5 or 0
    assert "mcnemar_p" not in seeds and seeds["pass@1_a"] == 0.375

    best = _per_task(tmp_path, "base", "1" * 40, split="kodcode_test_best_of_8")
    assert compare.compare([a], [best], split_b="kodcode_test_best_of_8")["only_b"] == 10

    with pytest.raises(SystemExit):
        compare.compare([a], [_per_task(tmp_path, "short", "1" * 5)])
