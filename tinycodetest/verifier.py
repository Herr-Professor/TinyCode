from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

from tinycodetest.harness import SandboxConfig
from tinycodetest.schema import Task, VerificationResult

_RESULT_PREFIX = "__TINYCODETEST_RESULT__"


def _runner_source(payload: dict[str, object]) -> str:
    blob = json.dumps(payload, separators=(",", ":"))
    return f"""
import contextlib
import io
import json
import traceback

RESULT_PREFIX = {_RESULT_PREFIX!r}
PAYLOAD = json.loads({blob!r})


def finish(data):
    print(RESULT_PREFIX + json.dumps(data, separators=(",", ":")))


def main():
    namespace = {{}}
    completion = PAYLOAD["completion"]
    try:
        exec(completion, namespace, namespace)
    except Exception as exc:
        finish({{
            "passed": False,
            "passed_cases": 0,
            "total_cases": len(PAYLOAD["tests"]),
            "reward": 0.0,
            "error": f"compile_error: {{exc}}",
            "failures": [{{"idx": -1, "reason": traceback.format_exc(limit=2)}}],
        }})
        return

    fn_name = PAYLOAD["function_name"]
    fn = namespace.get(fn_name)
    if not callable(fn):
        finish({{
            "passed": False,
            "passed_cases": 0,
            "total_cases": len(PAYLOAD["tests"]),
            "reward": 0.0,
            "error": f"missing_function: {{fn_name}}",
            "failures": [{{"idx": -1, "reason": f"Function `{{fn_name}}` not found."}}],
        }})
        return

    passed_cases = 0
    total_cases = len(PAYLOAD["tests"])
    weighted = 0.0
    total_weight = 0.0
    failures = []

    for idx, case in enumerate(PAYLOAD["tests"]):
        args = case.get("args", [])
        kwargs = case.get("kwargs", {{}})
        expected = case.get("expected")
        weight = float(case.get("weight", 1.0))
        total_weight += weight

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                actual = fn(*args, **kwargs)
        except Exception as exc:
            failures.append({{"idx": idx, "reason": f"runtime_error: {{exc}}"}})
            continue

        if actual == expected:
            passed_cases += 1
            weighted += weight
        else:
            failures.append(
                {{
                    "idx": idx,
                    "reason": "assertion_failed",
                    "expected": expected,
                    "actual": actual,
                }}
            )

    reward = 0.0
    if total_weight > 0:
        reward = weighted / total_weight

    finish({{
        "passed": passed_cases == total_cases,
        "passed_cases": passed_cases,
        "total_cases": total_cases,
        "reward": reward,
        "error": None if passed_cases == total_cases else "verification_failed",
        "failures": failures,
    }})


if __name__ == "__main__":
    main()
""".strip()


def _resource_limiter(memory_mb: int, timeout_seconds: float) -> Callable[[], None]:
    def _limit() -> None:
        try:
            import resource

            memory_bytes = int(memory_mb * 1024 * 1024)
            if hasattr(resource, "RLIMIT_AS"):
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

            cpu_seconds = max(1, math.ceil(timeout_seconds))
            if hasattr(resource, "RLIMIT_CPU"):
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        except Exception:
            return

    return _limit


def verify_completion(
    task: Task,
    completion: str,
    *,
    sandbox: SandboxConfig | None = None,
) -> VerificationResult:
    sandbox_cfg = sandbox or SandboxConfig()
    payload = {
        "completion": completion,
        "function_name": task.function_name,
        "tests": [case.to_dict() for case in task.test_cases],
    }

    with tempfile.TemporaryDirectory(prefix="tinycodetest_") as tempdir:
        runner_path = Path(tempdir) / "runner.py"
        runner_path.write_text(_runner_source(payload), encoding="utf-8")

        started = time.perf_counter()
        try:
            process = subprocess.run(
                [sys.executable, "-I", str(runner_path)],
                cwd=tempdir,
                capture_output=True,
                text=True,
                timeout=sandbox_cfg.timeout_seconds + 0.2,
                preexec_fn=_resource_limiter(sandbox_cfg.memory_mb, sandbox_cfg.timeout_seconds),
            )
        except subprocess.TimeoutExpired:
            runtime_ms = (time.perf_counter() - started) * 1000.0
            return VerificationResult(
                task_id=task.task_id,
                passed=False,
                passed_cases=0,
                total_cases=len(task.test_cases),
                reward=0.0,
                timed_out=True,
                runtime_ms=runtime_ms,
                error="timeout",
                failures=[{"idx": -1, "reason": "timeout"}],
            )

        runtime_ms = (time.perf_counter() - started) * 1000.0
        stdout_lines = process.stdout.splitlines()
        payload_line = None
        for line in reversed(stdout_lines):
            if line.startswith(_RESULT_PREFIX):
                payload_line = line[len(_RESULT_PREFIX) :]
                break

        if payload_line is None:
            err = process.stderr.strip() or "runner_failed_without_result"
            return VerificationResult(
                task_id=task.task_id,
                passed=False,
                passed_cases=0,
                total_cases=len(task.test_cases),
                reward=0.0,
                timed_out=False,
                runtime_ms=runtime_ms,
                error=err,
                failures=[{"idx": -1, "reason": err}],
            )

        try:
            raw = json.loads(payload_line)
        except json.JSONDecodeError as exc:
            return VerificationResult(
                task_id=task.task_id,
                passed=False,
                passed_cases=0,
                total_cases=len(task.test_cases),
                reward=0.0,
                timed_out=False,
                runtime_ms=runtime_ms,
                error=f"bad_verifier_payload: {exc}",
                failures=[{"idx": -1, "reason": str(exc)}],
            )

        return VerificationResult(
            task_id=task.task_id,
            passed=bool(raw.get("passed", False)),
            passed_cases=int(raw.get("passed_cases", 0)),
            total_cases=int(raw.get("total_cases", len(task.test_cases))),
            reward=float(raw.get("reward", 0.0)),
            timed_out=False,
            runtime_ms=runtime_ms,
            error=raw.get("error"),
            failures=raw.get("failures", []),
        )
