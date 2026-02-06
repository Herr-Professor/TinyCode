from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TestCase:
    args: list[Any]
    kwargs: dict[str, Any]
    expected: Any
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "args": self.args,
            "kwargs": self.kwargs,
            "expected": self.expected,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TestCase":
        return cls(
            args=payload.get("args", []),
            kwargs=payload.get("kwargs", {}),
            expected=payload.get("expected"),
            weight=float(payload.get("weight", 1.0)),
        )


@dataclass(frozen=True)
class Task:
    task_id: str
    difficulty: str
    prompt: str
    function_name: str
    signature: str
    canonical_solution: str
    test_cases: list[TestCase]
    tags: list[str]
    adversarial: bool = False
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "prompt": self.prompt,
            "function_name": self.function_name,
            "signature": self.signature,
            "canonical_solution": self.canonical_solution,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "tags": self.tags,
            "adversarial": self.adversarial,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Task":
        return cls(
            task_id=payload["task_id"],
            difficulty=payload["difficulty"],
            prompt=payload["prompt"],
            function_name=payload["function_name"],
            signature=payload["signature"],
            canonical_solution=payload["canonical_solution"],
            test_cases=[TestCase.from_dict(case) for case in payload.get("test_cases", [])],
            tags=list(payload.get("tags", [])),
            adversarial=bool(payload.get("adversarial", False)),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class VerificationResult:
    task_id: str
    passed: bool
    passed_cases: int
    total_cases: int
    reward: float
    timed_out: bool
    runtime_ms: float
    error: str | None = None
    failures: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "passed_cases": self.passed_cases,
            "total_cases": self.total_cases,
            "reward": self.reward,
            "timed_out": self.timed_out,
            "runtime_ms": self.runtime_ms,
            "error": self.error,
            "failures": self.failures or [],
        }
