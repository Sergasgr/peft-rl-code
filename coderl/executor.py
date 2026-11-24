"""Run a candidate solution against its pytest tests in a separate, resource-limited process.

The solution is written to `solution.py` and the tests to `test_solution.py`, which import it with
`from solution import *`. This is how KodCode's tests are meant to run (see code-r1).

This is not a security sandbox: the code runs as your user, with network access. Run the experiments
inside a container if you don't want model-written code on your machine.
"""

import os
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

TIMEOUT_S = 10
MEMORY_BYTES = 4 * 1024**3
FILE_SIZE_BYTES = 16 * 1024**2
WORKERS = min(8, os.cpu_count() or 1)

PYTEST_ARGS = ["-q", "-x", "--no-header", "-p", "no:cacheprovider", "-o", "addopts="]

# Sets the limits inside the child and then becomes pytest. (subprocess's preexec_fn would do the same, but it
# isn't safe when called from several threads, as run_many does.)
_LAUNCHER = """\
import os, resource, sys
memory, file_size = int(sys.argv[1]), int(sys.argv[2])
resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
resource.setrlimit(resource.RLIMIT_FSIZE, (file_size, file_size))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
os.execv(sys.executable, [sys.executable, "-I", "-m", "pytest", *sys.argv[3:]])
"""


@dataclass(frozen=True)
class ExecResult:
    passed: bool
    status: str  # "passed", "failed", "timeout" or "no_tests"
    output: str = ""


def run_tests(code: str, tests: str, timeout: float = TIMEOUT_S) -> ExecResult:
    with tempfile.TemporaryDirectory(prefix="coderl-") as tmp:
        with open(os.path.join(tmp, "solution.py"), "w", encoding="utf-8") as f:
            f.write(code)
        with open(os.path.join(tmp, "test_solution.py"), "w", encoding="utf-8") as f:
            f.write(tests)
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": tmp,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        command = [
            sys.executable,
            "-I",
            "-c",
            _LAUNCHER,
            str(MEMORY_BYTES),
            str(FILE_SIZE_BYTES),
            *PYTEST_ARGS,
            "test_solution.py",
        ]
        proc = subprocess.Popen(
            command,
            cwd=tmp,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group, so a timeout also kills what the solution spawned
        )
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            return ExecResult(False, "timeout")
    output = out.decode("utf-8", errors="replace")[-2000:]
    if proc.returncode == 0:
        return ExecResult(True, "passed", output)
    if proc.returncode == 5:  # pytest: no tests collected
        return ExecResult(False, "no_tests", output)
    return ExecResult(False, "failed", output)


def run_many(codes: list[str], tests: list[str], workers: int = WORKERS) -> list[ExecResult]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_tests, codes, tests))
