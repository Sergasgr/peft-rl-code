"""Train one experiment with GRPO, evaluate it and write its result JSON.

    uv run python run.py experiments/lora/lr1e-5-tests

The experiment folder holds adapter_config.json (none means full fine-tuning) and, optionally,
training_params.json with the settings that differ from default_training_params.json.
"""

import argparse
import dataclasses
import json
import platform
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

from coderl import rewards
from coderl.config import experiment_name, git_state, load_params, load_peft_config, output_root, result_path
from coderl.data import load_split
from coderl.evalplus import ScoringError
from coderl.evaluation import evaluate_evalplus, evaluate_kodcode
from coderl.forgetting import load_texts, mean_token_loss

PACKAGES = ("torch", "transformers", "peft", "trl", "datasets", "ruff", "lizard", "pylint", "radon")


def load_model(params: dict, device: str):
    kwargs = {
        "revision": params["model_revision"],
        "dtype": getattr(torch, params["dtype"]),
        "attn_implementation": params["attn_implementation"],
    }
    tokenizer = AutoTokenizer.from_pretrained(params["model_id"], revision=params["model_revision"])
    model = AutoModelForCausalLM.from_pretrained(params["model_id"], **kwargs).to(device)
    return model, tokenizer


def model_sha(params: dict) -> str | None:
    if Path(params["model_id"]).exists():
        return None
    from huggingface_hub import model_info

    try:
        return model_info(params["model_id"], revision=params["model_revision"]).sha
    except Exception:  # offline: the result still records model_id and model_revision
        return None


def jsonable(obj) -> object:
    return json.loads(json.dumps(obj, default=lambda o: sorted(o) if isinstance(o, set) else str(o)))


def grpo_config(params: dict, output_dir: Path, weights: list[float]) -> GRPOConfig:
    completions_per_step = params["prompts_per_step"] * params["num_generations"]
    if completions_per_step % params["micro_batch_size"]:
        raise SystemExit("prompts_per_step × num_generations must be a multiple of micro_batch_size")
    # transformers renamed gradient_checkpointing to activation_checkpointing; use whichever this version has
    checkpointing = (
        "activation_checkpointing"
        if "activation_checkpointing" in {f.name for f in dataclasses.fields(GRPOConfig)}
        else "gradient_checkpointing"
    )
    return GRPOConfig(
        output_dir=str(output_dir),
        **{checkpointing: params["gradient_checkpointing"]},
        per_device_train_batch_size=params["micro_batch_size"],
        gradient_accumulation_steps=completions_per_step // params["micro_batch_size"],
        num_generations=params["num_generations"],
        max_completion_length=params["max_completion_length"],
        temperature=params["temperature"],
        beta=params["beta"],  # 0: no reference model (with PiSSA, "adapter disabled" wouldn't be the base model)
        loss_type=params["loss_type"],
        mask_truncated_completions=params["mask_truncated_completions"],
        reward_weights=weights,
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        optim=params["optim"],
        lr_scheduler_type=params["lr_scheduler_type"],
        warmup_steps=params["warmup_steps"],
        max_grad_norm=params["max_grad_norm"],
        max_steps=params["max_steps"],
        bf16=params["dtype"] == "bfloat16",
        use_vllm=params["use_vllm"],
        logging_steps=params["logging_steps"],
        save_strategy="no",
        report_to="none",
        seed=params["seed"],
    )


def per_task(rows: list[dict]) -> str:
    """Pass/fail per task as a string of 0/1 in file order: enough for paired tests between runs."""
    return "".join("1" if r["passed"] else "0" for r in rows)


def evaluate(model, tokenizer, params: dict, data_dir: Path, workdir: Path) -> dict:
    limit = params["eval_limit"]
    out = {}
    for split in ("valid", "test"):
        summary, rows = evaluate_kodcode(model, tokenizer, load_split(data_dir, split)[:limit])
        out[f"kodcode_{split}"] = {**summary, "per_task": per_task(rows)}
    if params["evalplus"]:
        for name in ("humaneval", "mbpp"):
            try:
                summary, rows = evaluate_evalplus(model, tokenizer, name, workdir / name)
            except ScoringError as e:  # keep the run's other numbers; the table shows "—" for this benchmark
                print(f"{name}: {e}")
                out[name] = {"error": str(e)}
                continue
            out[name] = {**summary, "per_task": per_task(rows)}
    return out


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/kodcode"))
    args = parser.parse_args(argv)

    started = time.time()
    exp_dir = args.experiment
    name = experiment_name(exp_dir)
    params = load_params(exp_dir)
    peft_config = load_peft_config(exp_dir)
    git = git_state()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    workdir = output_root() / "runs" / name.replace("/", "--")
    print(f"{name}: {'full fine-tuning' if peft_config is None else peft_config.peft_type}, {params}")

    set_seed(params["seed"])
    model, tokenizer = load_model(params, device)
    train = load_split(args.data_dir, "train")
    forgetting = {}
    if params["forgetting"]:
        texts, wiki_sha = load_texts()
        forgetting = {"finewiki_sha": wiki_sha, "before": mean_token_loss(model, tokenizer, texts)}  # before wrapping

    funcs, weights = rewards.build(params["reward"], params["quality_weight"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=funcs,
        args=grpo_config(params, workdir / "trainer", weights),
        train_dataset=Dataset.from_list(train),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    model = trainer.model
    status, error, train_time = "success", "", None
    try:
        train_time = trainer.train().metrics["train_runtime"]
    except (KeyboardInterrupt, torch.OutOfMemoryError) as e:
        status, error = "cancelled", f"{type(e).__name__}: {e}"
    peak_memory = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None

    evaluation = {}
    if status == "success":
        if peft_config is not None:  # saved first, so a failed evaluation doesn't lose the training
            model.save_pretrained(output_root() / "checkpoints" / name.replace("/", "--"))
        evaluation = evaluate(model, tokenizer, params, args.data_dir, workdir)
        if params["forgetting"]:
            forgetting["after"] = mean_token_loss(model, tokenizer, texts)
            forgetting["delta"] = forgetting["after"] - forgetting["before"]

    result = {
        "run_info": {
            "experiment_name": name,
            "status": status,
            "error_msg": error,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "total_time": time.time() - started,
            "git": git,
            "params": params,
            "peft_config": None if peft_config is None else jsonable(peft_config.to_dict()),
        },
        "train_info": {
            "train_time": train_time,
            "num_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "num_total_params": sum(p.numel() for p in model.parameters()),
            "accelerator_memory_max": peak_memory,
            "log_history": trainer.state.log_history,
        },
        "eval": evaluation,
        "forgetting": forgetting,
        "meta_info": {
            "model_sha": model_sha(params),
            "data_manifest": json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8")),
            "packages": {p: version(p) for p in PACKAGES},
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
    }
    path = result_path(name, status, git)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Result: {path}")
    return path


if __name__ == "__main__":
    main()
