"""Experiment settings and where each run's result goes (the layout of peft's method_comparison)."""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = ROOT / "default_training_params.json"


def output_root() -> Path:
    return Path(os.environ.get("CODERL_OUTPUT", ROOT))


def load_params(exp_dir: Path, defaults: Path = DEFAULTS) -> dict:
    """default_training_params.json, overridden key by key by the experiment's training_params.json.
    Every setting is a flat key, so overriding one can't silently drop a sibling."""
    params = json.loads(defaults.read_text(encoding="utf-8"))
    params.pop("_comment", None)
    path = exp_dir / "training_params.json"
    if path.exists():
        override = json.loads(path.read_text(encoding="utf-8"))
        unknown = sorted(set(override) - set(params))
        if unknown:
            raise SystemExit(f"{path}: unknown settings {unknown}")
        params.update(override)
    return params


def load_peft_config(exp_dir: Path):
    """The experiment's adapter_config.json as a PeftConfig, or None for full fine-tuning."""
    if not (exp_dir / "adapter_config.json").exists():
        return None
    from peft import PeftConfig

    return PeftConfig.from_pretrained(str(exp_dir))


def experiment_name(exp_dir: Path) -> str:
    """experiments/lora/lr1e-5-tests → lora/lr1e-5-tests"""
    parts = exp_dir.resolve().parts
    if "experiments" not in parts:
        raise SystemExit(f"{exp_dir} is not inside an experiments/ folder")
    return "/".join(parts[len(parts) - parts[::-1].index("experiments") :])


def git_state(path: Path = ROOT) -> dict:
    """Branch, commit and whether tracked files changed (results/ and queue.txt don't count)."""

    def git(*args):
        return subprocess.run(["git", *args], cwd=path, capture_output=True, text=True).stdout.strip()

    try:
        branch, sha = git("rev-parse", "--abbrev-ref", "HEAD"), git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", ":!results", ":!queue.txt"))
    except FileNotFoundError:  # no git
        branch, sha, dirty = "", "", True
    return {"branch": branch or None, "sha": sha or None, "dirty": dirty}


def result_path(name: str, status: str, git: dict) -> Path:
    stem = name.replace("/", "--")
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if status != "success":
        return output_root() / "cancelled_results" / f"{stem}--{stamp}.json"
    if git["branch"] == "main" and not git["dirty"]:
        return output_root() / "results" / f"{stem}.json"
    return output_root() / "temporary_results" / f"{stem}--{stamp}.json"
