from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tinycodetest.schema import Task


class PromptStrategy(str, Enum):
    DIRECT = "direct"
    PLAN_THEN_CODE = "plan_then_code"


@dataclass(frozen=True)
class SandboxConfig:
    timeout_seconds: float = 2.0
    memory_mb: int = 256

    @classmethod
    def for_environment(cls, *, serverless: bool = False) -> "SandboxConfig":
        if serverless:
            return cls(timeout_seconds=0.6, memory_mb=256)
        return cls(timeout_seconds=2.0, memory_mb=256)


@dataclass(frozen=True)
class HarnessRequest:
    task: Task
    strategy: PromptStrategy = PromptStrategy.DIRECT


class PromptHarness:
    def __init__(self, sandbox: SandboxConfig | None = None) -> None:
        self.sandbox = sandbox or SandboxConfig()

    def build_prompt(self, request: HarnessRequest) -> str:
        if request.strategy == PromptStrategy.PLAN_THEN_CODE:
            preface = (
                "Think through edge cases briefly, then output only Python code for the required function. "
                "No prose in the final answer."
            )
        else:
            preface = "Output only Python code for the required function. No prose."

        return (
            f"{preface}\\n\\n"
            f"{request.task.prompt}\\n\\n"
            "Hard constraints:\\n"
            f"- Implement exactly `{request.task.function_name}` with this signature: {request.task.signature}\\n"
            "- Do not read from stdin or write to files.\\n"
            "- Avoid top-level side effects and prints."
        )

    def build_messages(self, request: HarnessRequest) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a coding model producing deterministic Python function implementations. "
                    "Return code only."
                ),
            },
            {"role": "user", "content": self.build_prompt(request)},
        ]
