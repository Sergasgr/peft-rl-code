"""Paired comparison of two configurations on the same tasks.

    uv run python compare.py --a results/lora--lr1e-5-tests.json --b results/dora--lr1e-5-tests.json
    uv run python compare.py --a results/lora--lr1e-5-tests.json results/lora--lr1e-5-tests-seed*.json \
        --b results/dora--lr1e-5-tests.json results/dora--lr1e-5-tests-seed*.json --split humaneval
    uv run python compare.py --a results/lora--lr1e-5-tests+quality.json \
        --b results/baseline--Qwen2.5-Coder-1.5B-Instruct.json --split-b kodcode_test_best_of_8

With one file per side: exact McNemar on the tasks only one side solves. With several (seeds): each task's pass
rate is averaged over the seeds and the difference gets a paired bootstrap interval, resampling tasks.
A difference is reported only when p < 0.05 or the interval excludes 0; otherwise "no detectable difference".
"""

import argparse
import json
from pathlib import Path

from coderl.stats import mcnemar_exact, paired_bootstrap


def task_rates(paths: list[Path], split: str) -> list[float]:
    """Per-task pass rate, averaged over the given runs (one run → 0 or 1 per task)."""
    runs = [json.loads(p.read_text(encoding="utf-8"))["eval"][split]["per_task"] for p in paths]
    if len({len(r) for r in runs}) != 1:
        raise SystemExit(f"{split}: the runs cover different numbers of tasks")
    return [sum(r[i] == "1" for r in runs) / len(runs) for i in range(len(runs[0]))]


def compare(a: list[Path], b: list[Path], split: str = "kodcode_test", split_b: str | None = None) -> dict:
    rate_a, rate_b = task_rates(a, split), task_rates(b, split_b or split)
    if len(rate_a) != len(rate_b):
        raise SystemExit("The two sides cover different numbers of tasks")
    n = len(rate_a)
    diff, low, high = paired_bootstrap(rate_a, rate_b)
    out = {
        "tasks": n,
        "pass@1_a": sum(rate_a) / n,
        "pass@1_b": sum(rate_b) / n,
        "diff": diff,
        "diff_ci": (low, high),
    }
    if len(a) == len(b) == 1:
        out["only_a"] = sum(x > y for x, y in zip(rate_a, rate_b, strict=True))
        out["only_b"] = sum(y > x for x, y in zip(rate_a, rate_b, strict=True))
        out["mcnemar_p"] = mcnemar_exact([x == 1 for x in rate_a], [y == 1 for y in rate_b])
    out["detectable"] = out.get("mcnemar_p", 1.0) < 0.05 or low > 0 or high < 0
    return out


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", type=Path, nargs="+", required=True, help="result files of configuration A (seeds)")
    parser.add_argument("--b", type=Path, nargs="+", required=True, help="result files of configuration B (seeds)")
    parser.add_argument("--split", default="kodcode_test", help="kodcode_test, kodcode_valid, humaneval or mbpp")
    parser.add_argument("--split-b", default=None, help="B's split if it differs, e.g. kodcode_test_best_of_8")
    args = parser.parse_args(argv)

    out = compare(args.a, args.b, args.split, args.split_b)
    print(f"{out['tasks']} tasks: A {100 * out['pass@1_a']:.1f}%  B {100 * out['pass@1_b']:.1f}%")
    low, high = out["diff_ci"]
    print(f"A − B: {100 * out['diff']:+.1f} points, 95% CI [{100 * low:+.1f}, {100 * high:+.1f}]")
    if "mcnemar_p" in out:
        print(f"only A solves {out['only_a']}, only B {out['only_b']}; exact McNemar p = {out['mcnemar_p']:.3g}")
    print("difference detectable" if out["detectable"] else "no detectable difference")
    return out


if __name__ == "__main__":
    main()
