"""Static quality measures for one Python solution.

Two groups, on purpose:
- in the reward (what RL optimizes): ruff violations and lizard's cyclomatic complexity, combined in quality_score();
- outside the reward (what RL never sees): pylint's score and radon's maintainability index.
A quality gain that shows up only in the first group is the reward being optimized, not better code.
"""

import ast
import json
import os
import subprocess
import sys
import tempfile

import lizard
from radon.metrics import mi_visit
from ruff.__main__ import find_ruff_bin

RUFF_RULES = "F,E9,B,SIM"  # pyflakes, syntax errors, bugbear, simplify
MAX_CC = 10
RUFF = find_ruff_bin()


def parses(code: str) -> bool:
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return bool(code.strip())


def _function(code: str, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def defines(code: str, name: str) -> bool:
    return _function(code, name) is not None


def trivial_body(code: str, name: str) -> bool:
    """The function only has a docstring, `pass`, `...` or `return <constant>`."""
    func = _function(code, name)
    if func is None:
        return False
    for stmt in func.body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):  # docstring or `...`
            continue
        if isinstance(stmt, ast.Return) and (stmt.value is None or isinstance(stmt.value, ast.Constant)):
            continue
        return False
    return True


def comment_ratio(code: str) -> float:
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    return sum(line.startswith("#") for line in lines) / len(lines) if lines else 0.0


def code_lines(code: str) -> int:
    return sum(1 for line in code.splitlines() if line.strip() and not line.strip().startswith("#"))


def is_noop(code: str, entry_point: str) -> bool:
    """Doesn't define the entry point, is ≥ 90% comment lines, or the entry point does nothing."""
    return not defines(code, entry_point) or comment_ratio(code) >= 0.9 or trivial_body(code, entry_point)


def ruff_violations(code: str) -> int:
    result = subprocess.run(
        [
            RUFF,
            "check",
            "--isolated",
            "--no-cache",
            "--quiet",
            "--select",
            RUFF_RULES,
            "--target-version",
            "py311",
            "--output-format",
            "json",
            "--stdin-filename",
            "solution.py",
            "-",
        ],
        input=code,
        capture_output=True,
        text=True,
        timeout=30,
    )  # exit code 1 means "violations found", so no check=True
    return len(json.loads(result.stdout or "[]"))


def max_cyclomatic_complexity(code: str) -> int:
    functions = lizard.analyze_file.analyze_source_code("solution.py", code).function_list
    return max((f.cyclomatic_complexity for f in functions), default=0)


def quality_score(code: str, entry_point: str, ref_cc: int) -> float:
    """The quality term of the reward, in [0, 1]: half lint, half complexity relative to the reference solution.
    0 if the code doesn't parse or doesn't define the entry point; nothing else stops it from being gamed."""
    if not parses(code) or not defines(code, entry_point):
        return 0.0
    lint = 1.0 / (1 + ruff_violations(code))
    complexity = min(1.0, max(ref_cc, 1) / max(max_cyclomatic_complexity(code), 1))
    return 0.5 * lint + 0.5 * complexity


def pylint_score(code: str) -> float | None:
    if not parses(code):
        return None
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "solution.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pylint",
                "--output-format=json2",
                "--score=y",
                "--persistent=n",
                "--disable=import-error",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    try:
        return float(json.loads(result.stdout)["statistics"]["score"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def maintainability_index(code: str) -> float | None:
    if not parses(code):
        return None
    try:
        return float(mi_visit(code, multi=True))
    except Exception:  # radon raises on some valid but unusual code
        return None


def measure(code: str, entry_point: str, ref_cc: int, independent: bool = True) -> dict:
    ok = parses(code)
    row = {
        "parses": ok,
        "defines_entry_point": defines(code, entry_point),
        "noop": is_noop(code, entry_point),
        "code_lines": code_lines(code),
        "comment_ratio": comment_ratio(code),
        "ruff": ruff_violations(code) if ok else None,
        "cc": max_cyclomatic_complexity(code) if ok else None,
        "quality": quality_score(code, entry_point, ref_cc),
    }
    if independent:
        row["pylint"] = pylint_score(code)
        row["mi"] = maintainability_index(code)
    return row
