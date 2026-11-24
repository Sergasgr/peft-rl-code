"""HumanEval+ and MBPP+ with EvalPlus's own chat prompt and scorer.

Pinned on purpose:
- EvalPlus at commit 6eb1e19: release 0.3.1 never passes HumanEval/32, even with the canonical solution.
- Python 3.11 for scoring: Python 3.12's float sum() breaks HumanEval/32's oracle.
- EVALPLUS_MAX_MEMORY_BYTES = 8 GiB: the 4 GiB default fails Mbpp/255's canonical solution.
- The results file is deleted before scoring: EvalPlus silently reuses an old one.
- evalplus.evaluate runs through a wrapper that exits when it finishes: if a worker is killed (e.g. by the OOM
  killer), EvalPlus raises BrokenProcessPool and then hangs forever on a non-daemon thread. On MBPP+ the main
  process takes ~2 GB and each worker up to ~4 GB, so a failure with 2 workers is retried with 1. A timeout
  covers any other hang.

make_raw_chat_prompt and the stop strings are copied from EvalPlus (Apache-2.0,
https://github.com/evalplus/evalplus/blob/6eb1e199c2e370518e7bf3e8eee6322c2b23c89c/evalplus/provider/utility.py).
"""

import gzip
import hashlib
import json
import os
import shutil
import signal
import subprocess
import urllib.request
from pathlib import Path

EVALPLUS = "evalplus @ git+https://github.com/evalplus/evalplus@6eb1e199c2e370518e7bf3e8eee6322c2b23c89c"
SCORING_PYTHON = "3.11"
MAX_MEMORY_BYTES = 8 * 1024**3
PARALLEL = 2
TIMEOUT_S = 1800  # per EvalPlus call; scoring 378 solutions normally takes a few minutes

# `python -c` inside EvalPlus's environment: runs evalplus.evaluate and exits even if a non-daemon thread is left.
_EVALUATE = """
import os, sys, traceback
from evalplus.evaluate import main
sys.argv = ["evalplus.evaluate", *sys.argv[1:]]
try:
    main()
    code = 0
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else 1
except BaseException:
    traceback.print_exc()
    code = 1
sys.stdout.flush()
sys.stderr.flush()
os._exit(code)
"""
CONSTRAINTS = Path(__file__).resolve().parent.parent / "scoring-constraints.txt"
CACHE = Path(os.environ.get("CODERL_CACHE", Path.home() / ".cache" / "coderl"))

DATASETS = {
    "humaneval": (
        "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz",
        "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101",
        164,
    ),
    "mbpp": (
        "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz",
        "af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63",
        378,
    ),
}

INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"
)
STOP_STRINGS = ["<|endoftext|>", "<|endofmask|>", "</s>", "\nif __name__", "\ndef main(", "\nprint(", "\n```\n"]
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"


def make_raw_chat_prompt(task_prompt: str, tokenizer) -> str:
    task_prompt = f"""\
{INSTRUCTION_PREFIX}
```
{task_prompt.strip()}
```
"""
    response = f"""\
{RESPONSE_PREFIX}
```python
{_MAGIC_SPLITTER_}
```
"""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": task_prompt}, {"role": "assistant", "content": response}],
        tokenize=False,
    ).split(_MAGIC_SPLITTER_)[0]


def load_tasks(name: str) -> list[dict]:
    url, sha256, n = DATASETS[name]
    path = CACHE / Path(url).name
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise SystemExit(f"{path} doesn't have the pinned sha256; delete it and retry")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]
    assert len(tasks) == n, (name, len(tasks))
    return tasks


def _uvx(args: list[str]) -> list[str]:
    if shutil.which("uvx") is None:
        raise SystemExit("EvalPlus scoring needs `uvx` (https://docs.astral.sh/uv/)")
    return ["uvx", "--python", SCORING_PYTHON, "--constraints", str(CONSTRAINTS), "--from", EVALPLUS, *args]


class ScoringError(RuntimeError):
    pass


def _run(cmd: list[str], env: dict, timeout: float, label: str = "EvalPlus") -> None:
    """Run in its own process group, so a timeout also kills uvx's child Python and EvalPlus's workers."""
    proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        raise ScoringError(f"{label} timed out after {timeout:.0f} s") from None
    if code != 0:
        raise ScoringError(f"{label} exited with {code}")


def score(
    solutions: dict[str, str], name: str, workdir: Path, parallel: int = PARALLEL, timeout: float = TIMEOUT_S
) -> dict[str, dict]:
    """task_id → {"base": bool, "plus": bool, "sanitized": str}. `plus` means base and plus tests both pass.
    Raises ScoringError if EvalPlus fails twice (the second time with a single worker)."""
    workdir.mkdir(parents=True, exist_ok=True)
    raw = workdir / f"{name}.jsonl"
    raw.write_text(
        "".join(json.dumps({"task_id": t, "solution": s}) + "\n" for t, s in solutions.items()), encoding="utf-8"
    )
    sanitized = workdir / f"{name}-sanitized.jsonl"
    results = workdir / f"{name}-sanitized_eval_results.json"
    env = {**os.environ, "EVALPLUS_MAX_MEMORY_BYTES": str(MAX_MEMORY_BYTES)}
    _run(_uvx(["evalplus.sanitize", "--samples", str(raw)]), env, timeout, f"evalplus.sanitize {name}")
    for attempt, workers in enumerate(dict.fromkeys([parallel, 1])):
        results.unlink(missing_ok=True)
        cmd = ["python", "-c", _EVALUATE, "--dataset", name, "--samples", str(sanitized), "--parallel", str(workers)]
        try:
            _run(_uvx(cmd), env, timeout, f"evalplus.evaluate {name} with {workers} workers")
            break
        except ScoringError as e:
            if workers == 1:
                raise
            print(f"EvalPlus attempt {attempt + 1} failed ({e}); retrying with 1 worker")
    clean = {
        json.loads(line)["task_id"]: json.loads(line)["solution"]
        for line in sanitized.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    per_task = {}
    for task_id, [entry] in json.loads(results.read_text(encoding="utf-8"))["eval"].items():
        base = entry["base_status"] == "pass"
        per_task[task_id] = {"base": base, "plus": base and entry["plus_status"] == "pass", "sanitized": clean[task_id]}
    missing = set(solutions) - set(per_task)
    if missing:
        raise ScoringError(f"EvalPlus returned no result for {len(missing)} tasks, e.g. {sorted(missing)[:3]}")
    return per_task
