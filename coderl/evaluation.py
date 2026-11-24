"""Evaluation of a model, trained or not, on the KodCode splits and on HumanEval+ / MBPP+.

Greedy decoding with every setting written out: transformers fills any setting you leave unset from the model's
generation_config.json, and Qwen's asks for sampling with repetition_penalty 1.1.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

import torch
from transformers import GenerationConfig

from coderl import evalplus
from coderl.data import extract_code, render
from coderl.executor import WORKERS, run_many
from coderl.quality import MAX_CC, max_cyclomatic_complexity, measure, quality_score
from coderl.stats import wilson_ci

MAX_NEW_TOKENS = 768
BATCH_SIZE = 16


def stop_token_ids(tokenizer) -> list[int]:
    ids = {tokenizer.eos_token_id}
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
            ids.add(token_id)
    return sorted(i for i in ids if i is not None)


def generation_config(
    tokenizer,
    sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.95,
    stop_strings: list[str] | None = None,
) -> GenerationConfig:
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return GenerationConfig(
        do_sample=sample,
        temperature=temperature if sample else 1.0,
        top_p=top_p if sample else 1.0,
        top_k=0,
        num_beams=1,
        repetition_penalty=1.0,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
        eos_token_id=stop_token_ids(tokenizer),
        pad_token_id=pad,
        stop_strings=stop_strings,
    )


def _cut(text: str, stops: list[str] | None) -> str:
    end = min((text.find(s) for s in stops or [] if s in text), default=len(text))
    return text[:end].replace("\t", "    ")


@torch.inference_mode()
def generate(model, tokenizer, prompts: list[str], config: GenerationConfig, n: int = 1) -> list[list[str]]:
    """n replies per prompt. The config is also installed on the inner model, because with a PeftModel
    `model.generation_config = ...` stays on the wrapper and generate() would use the Hub one."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    saved_config, saved_side, was_training = inner.generation_config, tokenizer.padding_side, model.training
    inner.generation_config, tokenizer.padding_side = config, "left"
    model.eval()
    batch_size = max(1, BATCH_SIZE // n)
    replies = []
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            out = model.generate(**enc, generation_config=config, num_return_sequences=n, tokenizer=tokenizer)
            texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            replies.extend(
                [_cut(t, config.stop_strings) for t in texts[i * n : (i + 1) * n]] for i in range(len(batch))
            )
    finally:
        inner.generation_config, tokenizer.padding_side = saved_config, saved_side
        model.train(was_training)
    return replies


def _measure_all(codes: list[str], entry_points: list[str], ref_ccs: list[int]) -> list[dict]:
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(measure, codes, entry_points, ref_ccs))


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    passing = [r for r in rows if r["passed"]]
    clean = [r for r in passing if r["ruff"] == 0 and r["cc"] is not None and r["cc"] <= MAX_CC]

    def avg(key, subset):
        values = [r[key] for r in subset if r[key] is not None]
        return mean(values) if values else None

    return {
        "n": n,
        "pass@1": len(passing) / n,
        "pass@1_ci": wilson_ci(len(passing), n),
        "clean_pass@1": len(clean) / n,
        "clean_pass@1_ci": wilson_ci(len(clean), n),
        "passing": {key: avg(key, passing) for key in ("quality", "ruff", "cc", "pylint", "mi", "code_lines")},
        "noop_rate": mean(r["noop"] for r in rows),
        "comment_ratio": mean(r["comment_ratio"] for r in rows),
        "code_lines": mean(r["code_lines"] for r in rows),
        "parse_failure_rate": mean(not r["parses"] for r in rows),
    }


def evaluate_kodcode(model, tokenizer, records: list[dict], n: int = 1) -> tuple[dict, list[dict]]:
    """Greedy pass@1 when n == 1. With n > 1: sample n replies and keep the one with the best quality score,
    without running tests (the best-of-n baseline reranked by the analyzer, at the same sampling budget)."""
    prompts = [render(tokenizer, r["prompt"]) for r in records]
    replies = generate(model, tokenizer, prompts, generation_config(tokenizer, sample=n > 1), n=n)
    codes = []
    for record, candidates in zip(records, replies, strict=True):
        candidate_codes = [extract_code(c) for c in candidates]
        if n > 1:
            candidate_codes.sort(key=lambda c: quality_score(c, record["entry_point"], record["ref_cc"]), reverse=True)
        codes.append(candidate_codes[0])
    results = run_many(codes, [r["tests"] for r in records])
    metrics = _measure_all(codes, [r["entry_point"] for r in records], [r["ref_cc"] for r in records])
    rows = [
        {"id": r["id"], "passed": res.passed, "status": res.status, **m}
        for r, res, m in zip(records, results, metrics, strict=True)
    ]
    return summarize(rows), rows


def evaluate_evalplus(model, tokenizer, name: str, workdir: Path) -> tuple[dict, list[dict]]:
    tasks = evalplus.load_tasks(name)
    prompts = [evalplus.make_raw_chat_prompt(t["prompt"], tokenizer) for t in tasks]
    replies = generate(model, tokenizer, prompts, generation_config(tokenizer, stop_strings=evalplus.STOP_STRINGS))
    scored = evalplus.score({t["task_id"]: r[0] for t, r in zip(tasks, replies, strict=True)}, name, workdir)
    codes = [scored[t["task_id"]]["sanitized"] for t in tasks]
    ref_ccs = [max_cyclomatic_complexity(t["prompt"] + t["canonical_solution"]) for t in tasks]
    metrics = _measure_all(codes, [t["entry_point"] for t in tasks], ref_ccs)
    rows = [
        {"id": t["task_id"], "passed": scored[t["task_id"]]["plus"], "passed_base": scored[t["task_id"]]["base"], **m}
        for t, m in zip(tasks, metrics, strict=True)
    ]
    summary = summarize(rows)
    summary["pass@1_base"] = mean(r["passed_base"] for r in rows)
    return summary, rows
