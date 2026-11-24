"""Evaluate the base model without training: the starting point of every experiment, and the best-of-n baseline.

    uv run python baseline.py                      # greedy: KodCode valid/test, HumanEval+, MBPP+
    uv run python baseline.py --best-of-n 8        # also: 8 samples per KodCode test task, keep the cleanest

Best-of-n reranks by the reward's quality score without running tests, at the sampling budget of one GRPO group.
If RL's quality gains don't beat it, a reranker would have been enough.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from coderl.config import DEFAULTS, git_state, output_root
from coderl.data import load_split
from coderl.evaluation import evaluate_kodcode
from run import PACKAGES, evaluate, load_model, model_sha, per_task, save_rows


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/kodcode"))
    parser.add_argument("--model-id", default=None, help="default: model_id of default_training_params.json")
    parser.add_argument("--best-of-n", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=None, help="only the first N tasks per split (smoke tests)")
    parser.add_argument("--no-evalplus", action="store_true")
    args = parser.parse_args(argv)

    params = json.loads(DEFAULTS.read_text(encoding="utf-8"))
    params["eval_limit"] = args.eval_limit
    params["evalplus"] = not args.no_evalplus
    if args.model_id:
        params["model_id"] = args.model_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(params, device)
    tag = params["model_id"].rstrip("/").split("/")[-1]
    workdir = output_root() / "runs" / f"baseline--{tag}"

    evaluation = evaluate(model, tokenizer, params, args.data_dir, workdir)
    if args.best_of_n > 1:
        records = load_split(args.data_dir, "test")[: params["eval_limit"]]
        summary, rows = evaluate_kodcode(model, tokenizer, records, n=args.best_of_n)
        evaluation[f"kodcode_test_best_of_{args.best_of_n}"] = {**summary, "per_task": per_task(rows)}
        save_rows(workdir / f"kodcode_test_best_of_{args.best_of_n}.jsonl", rows)

    from importlib.metadata import version

    result = {
        "run_info": {
            "experiment_name": f"baseline/{tag}",
            "status": "success",
            "git": git_state(),
            "params": params,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "eval": evaluation,
        "meta_info": {
            "model_sha": model_sha(params),
            "packages": {p: version(p) for p in PACKAGES},
            "data_manifest": json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8")),
        },
    }
    path = output_root() / "results" / f"baseline--{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Result: {path}")
    return path


if __name__ == "__main__":
    main()
