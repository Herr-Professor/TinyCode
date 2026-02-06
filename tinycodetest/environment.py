from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tinycodetest.dataset import TaskSampler, load_dataset
from tinycodetest.harness import HarnessRequest, PromptHarness, PromptStrategy, SandboxConfig
from tinycodetest.schema import Task, VerificationResult
from tinycodetest.verifier import verify_completion


@dataclass
class TinyCodeTestEnvironment:
    dataset_path: str | Path
    sandbox: SandboxConfig = SandboxConfig()

    def __post_init__(self) -> None:
        self.tasks = load_dataset(self.dataset_path)
        self.sampler = TaskSampler(self.tasks)
        self.harness = PromptHarness(sandbox=self.sandbox)

    def sample(
        self,
        n: int,
        *,
        difficulty: str | None = None,
        adversarial_only: bool = False,
        seed: int = 0,
    ) -> list[Task]:
        return self.sampler.sample(
            n,
            difficulty=difficulty,
            adversarial_only=adversarial_only,
            seed=seed,
        )

    def prompt(self, task: Task, strategy: PromptStrategy = PromptStrategy.DIRECT) -> str:
        return self.harness.build_prompt(HarnessRequest(task=task, strategy=strategy))

    def verify(self, task: Task, completion: str) -> VerificationResult:
        return verify_completion(task, completion, sandbox=self.sandbox)
