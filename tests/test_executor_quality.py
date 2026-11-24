from coderl.executor import run_many, run_tests
from coderl.quality import (
    defines,
    is_noop,
    max_cyclomatic_complexity,
    measure,
    quality_score,
    ruff_violations,
    trivial_body,
)

TESTS = "from solution import *\n\ndef test_add():\n    assert add(2, 3) == 5\n"


def test_passing_and_failing_solutions():
    assert run_tests("def add(a, b):\n    return a + b\n", TESTS).status == "passed"
    assert run_tests("def add(a, b):\n    return a - b\n", TESTS).status == "failed"
    assert run_tests("def add(a, b) return a\n", TESTS).status == "failed"  # syntax error


def test_timeout_and_memory_limits():
    assert run_tests("def add(a, b):\n    while True:\n        pass\n", TESTS, timeout=2).status == "timeout"
    hog = "def add(a, b):\n    x = bytearray(8 * 1024**3)\n    return a + b\n"  # over the 4 GiB address-space limit
    assert run_tests(hog, TESTS).status == "failed"


def test_no_tests_is_not_a_pass():
    assert run_tests("def add(a, b):\n    return a + b\n", "from solution import *\n").status == "no_tests"


def test_run_many_keeps_order():
    good, bad = "def add(a, b):\n    return a + b\n", "def add(a, b):\n    return 0\n"
    assert [r.passed for r in run_many([good, bad, good], [TESTS] * 3, workers=3)] == [True, False, True]


def test_static_measures():
    assert ruff_violations("import os\n\ndef f():\n    return 1\n") == 1  # F401 unused import
    assert ruff_violations("def f():\n    return 1\n") == 0
    assert max_cyclomatic_complexity("def f(x):\n    if x:\n        return 1\n    for _ in x:\n        pass\n") == 3
    assert defines("class A:\n    def f(self):\n        pass\n", "f")


def test_noop_detection():
    assert is_noop("# def add(a, b):\n#     return a + b\n", "add")  # all comments
    assert is_noop("def plus(a, b):\n    return a + b\n", "add")  # wrong name
    assert is_noop('def add(a, b):\n    """Adds."""\n    return None\n', "add")  # does nothing
    assert trivial_body("def add(a, b):\n    ...\n", "add")
    assert not is_noop("def add(a, b):\n    return a + b\n", "add")


def test_quality_score_rewards_lint_and_relative_complexity():
    clean = "def add(a, b):\n    return a + b\n"
    assert quality_score(clean, "add", ref_cc=1) == 1.0
    assert quality_score("import os\n" + clean, "add", ref_cc=1) == 0.5 * 0.5 + 0.5  # one violation
    twice_as_complex = "def add(a, b):\n    if a:\n        return a + b\n    return b\n"
    assert quality_score(twice_as_complex, "add", ref_cc=1) == 0.5 + 0.5 * 0.5
    assert quality_score("def add(a, b) return", "add", ref_cc=1) == 0.0  # doesn't parse
    assert quality_score(clean, "sub", ref_cc=1) == 0.0  # no entry point


def test_quality_score_can_be_gamed_on_purpose():
    """The `quality` reward has no correctness check: a do-nothing function scores the maximum."""
    assert quality_score("def add(a, b):\n    return 0\n", "add", ref_cc=3) == 1.0


def test_measure_includes_independent_metrics():
    row = measure("def add(a, b):\n    return a + b\n", "add", ref_cc=1)
    assert row["ruff"] == 0 and row["cc"] == 1 and not row["noop"]
    assert 0 <= row["pylint"] <= 10 and 0 <= row["mi"] <= 100
