"""Confidence intervals and paired tests for pass rates (from CodeAlign's report_utils.py)."""

import math
import random


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(a: list[bool], b: list[bool]) -> float:
    """Two-sided exact McNemar p-value for paired pass/fail results on the same tasks."""
    if len(a) != len(b):
        raise ValueError("Both runs must cover the same tasks")
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    n = only_a + only_b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n)


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of a − b over tasks and its 95% interval, resampling tasks. Use per-task means over seeds as input."""
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n_boot))
    return sum(diffs) / len(diffs), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
