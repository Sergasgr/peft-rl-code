"""GRPO reward functions.

TRL calls each function with `prompts`, `completions` and every other dataset column (`tests`, `entry_point`,
`ref_cc`, ...) as lists with one entry per completion, plus `log_metric`. Three reward modes:

    tests          1 if all unit tests pass, else 0
    tests+quality  the above + quality_weight · quality, where quality only counts when the tests pass
    quality        quality only: the probe for reward hacking (nothing checks correctness)

`monitor` has weight 0: it changes no reward and logs what the reward may not see (pass rate in `quality` runs,
no-op rate, code length, comments).
"""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from statistics import mean

from coderl.data import extract_code
from coderl.executor import WORKERS, run_tests
from coderl.quality import code_lines, comment_ratio, is_noop, quality_score


def _text(completion) -> str:
    return completion[0]["content"] if isinstance(completion, list) else completion


def _codes(completions) -> list[str]:
    return [extract_code(_text(c)) for c in completions]


@lru_cache(maxsize=16384)
def _passed(code: str, tests: str) -> bool:
    return run_tests(code, tests).passed


@lru_cache(maxsize=16384)
def _quality(code: str, entry_point: str, ref_cc: int) -> float:
    return quality_score(code, entry_point, ref_cc)


def _parallel(fn, *columns) -> list:
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(fn, *columns))


def tests_reward(completions, tests, **kwargs) -> list[float]:
    return [float(p) for p in _parallel(_passed, _codes(completions), tests)]


def gated_quality_reward(completions, tests, entry_point, ref_cc, **kwargs) -> list[float]:
    codes = _codes(completions)
    passed = _parallel(_passed, codes, tests)
    quality = _parallel(_quality, codes, entry_point, ref_cc)
    return [q if p else 0.0 for p, q in zip(passed, quality, strict=True)]


def quality_reward(completions, entry_point, ref_cc, **kwargs) -> list[float]:
    return _parallel(_quality, _codes(completions), entry_point, ref_cc)


def monitor(completions, tests, entry_point, log_metric=None, **kwargs) -> list[float]:
    codes = _codes(completions)
    if log_metric is not None:
        log_metric("monitor/pass_rate", mean(_parallel(_passed, codes, tests)))
        log_metric("monitor/noop_rate", mean(is_noop(c, e) for c, e in zip(codes, entry_point, strict=True)))
        log_metric("monitor/code_lines", mean(code_lines(c) for c in codes))
        log_metric("monitor/comment_ratio", mean(comment_ratio(c) for c in codes))
        log_metric("monitor/no_code_block", mean("```" not in _text(c) for c in completions))
    return [0.0] * len(codes)


def build(mode: str, quality_weight: float) -> tuple[list, list[float]]:
    """Reward functions and their weights for GRPOConfig(reward_weights=...)."""
    if mode == "tests":
        return [tests_reward, monitor], [1.0, 0.0]
    if mode == "tests+quality":
        return [tests_reward, gated_quality_reward, monitor], [1.0, quality_weight, 0.0]
    if mode == "quality":
        return [quality_reward, monitor], [1.0, 0.0]
    raise ValueError(f"Unknown reward mode {mode!r}: use 'tests', 'tests+quality' or 'quality'")
