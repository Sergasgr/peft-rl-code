"""KodCode problems with unit tests: filtering, validation, fixed splits, and code extraction from replies."""

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from coderl.executor import run_many
from coderl.quality import max_cyclomatic_complexity, ruff_violations

SOURCE = "KodCode/KodCode-Light-RL-10K"
SPLITS = ("train", "valid", "test")

# Libraries that aren't installed or shouldn't run in the executor (the list code-r1 uses for this dataset).
BLOCKED = (
    "fake-useragent",
    "keras",
    "socket",
    "torch",
    "scipy",
    "sklearn",
    "cv2",
    "imageio",
    "sphinx-pyproject",
    "xgboost",
    "tweepy",
    "flask",
    "matplotlib",
    "pillow",
    "seaborn",
    "smtpd",
    "pandas",
    "bs4",
)

PROMPT = (
    "Please solve the programming task below in Python.\n\n{question}\n\n"
    "Note that the function declaration is {declaration}. "
    "Return only the code, in a single markdown code block."
)

_FENCE = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)


def extract_code(reply: str) -> str:
    """The first markdown code block (also if the reply was cut before the closing fence), else the whole reply."""
    match = _FENCE.search(reply)
    return (match.group(1) if match else reply).strip("\n")


def build_prompt(question: str, declaration: str) -> str:
    return PROMPT.format(question=question.strip(), declaration=declaration.strip())


def render(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def prompt_tokens(tokenizer, messages: list[dict]) -> int:
    return len(tokenizer(render(tokenizer, messages), add_special_tokens=False)["input_ids"])


def _load_source(source: str, revision: str | None) -> tuple[list[dict], str]:
    path = Path(source)
    if path.exists():
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            rows = pq.read_table(path).to_pylist()
        else:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return rows, "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    from datasets import load_dataset
    from huggingface_hub import dataset_info

    sha = dataset_info(source, revision=revision).sha
    return list(load_dataset(source, split="train", revision=sha)), sha


def _candidate(row: dict) -> dict | None:
    """A function task whose solution and tests use no blocked library, or None."""
    info = row.get("test_info") or []
    tests = row.get("test") or row.get("test_code") or ""
    if not info or not info[0].get("function_name") or not info[0].get("function_declaration") or not tests.strip():
        return None
    if any(lib in row["solution"] or lib in tests for lib in BLOCKED):
        return None
    return {
        "id": row["question_id"],
        "prompt": [{"role": "user", "content": build_prompt(row["question"], info[0]["function_declaration"])}],
        "tests": "from solution import *\n" + tests.strip() + "\n",
        "entry_point": info[0]["function_name"],
        "solution": row["solution"],
    }


def prepare(
    out_dir: Path,
    source: str = SOURCE,
    revision: str | None = None,
    n_valid: int = 200,
    n_test: int = 500,
    seed: int = 0,
    tokenizer=None,
    max_prompt_tokens: int = 1024,
) -> dict:
    rows, resolved = _load_source(source, revision)
    counts = Counter(source_rows=len(rows))
    candidates = [c for c in map(_candidate, rows) if c is not None]
    counts["function_tasks_without_blocked_libraries"] = len(candidates)
    if tokenizer is not None:  # TRL doesn't truncate prompts, so long ones are dropped here
        candidates = [c for c in candidates if prompt_tokens(tokenizer, c["prompt"]) <= max_prompt_tokens]
        counts[f"prompt_at_most_{max_prompt_tokens}_tokens"] = len(candidates)
    results = run_many([c["solution"] for c in candidates], [c["tests"] for c in candidates])
    candidates = [c for c, r in zip(candidates, results, strict=True) if r.passed]
    counts["reference_solution_passes"] = len(candidates)
    if len(candidates) < n_valid + n_test + 1:
        raise SystemExit(f"Only {len(candidates)} usable problems, fewer than valid + test + 1")

    for c in candidates:
        solution = c.pop("solution")
        c["ref_cc"] = max_cyclomatic_complexity(solution)
        c["ref_ruff"] = ruff_violations(solution)
    random.Random(seed).shuffle(candidates)
    splits = {
        "test": candidates[:n_test],
        "valid": candidates[n_test : n_test + n_valid],
        "train": candidates[n_test + n_valid :],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, records in splits.items():
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        (out_dir / f"{name}.jsonl").write_text(text, encoding="utf-8")
        files[name] = {"rows": len(records), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    manifest = {"source": source, "revision": resolved, "seed": seed, "filters": dict(counts), "files": files}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_split(data_dir: Path, name: str) -> list[dict]:
    """Loads a split and checks it against the manifest, so a run can't silently use different data."""
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    text = (data_dir / f"{name}.jsonl").read_text(encoding="utf-8")
    if hashlib.sha256(text.encode()).hexdigest() != manifest["files"][name]["sha256"]:
        raise SystemExit(f"{data_dir / name}.jsonl doesn't match manifest.json; rerun `make data`")
    return [json.loads(line) for line in text.splitlines() if line.strip()]
