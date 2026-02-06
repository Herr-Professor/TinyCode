"""Optional Verifiers integration for TinyCodeTest.

This module intentionally keeps `verifiers` as an optional dependency.
If Verifiers is not installed, importing TinyCode core still works.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinycodetest.environment import TinyCodeTestEnvironment
from tinycodetest.harness import PromptStrategy, SandboxConfig

try:
    from verifiers import Episode, SingleTurnEnv

    VERIFIERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    Episode = Any  # type: ignore[assignment]
    SingleTurnEnv = object  # type: ignore[assignment]
    VERIFIERS_AVAILABLE = False


def _build_episode(prompt: str, metadata: dict[str, Any]) -> Any:
    """Best-effort Episode constructor compatibility across Verifiers versions."""
    if not VERIFIERS_AVAILABLE:
        raise RuntimeError("verifiers is not installed.")
    try:
        return Episode(prompt=prompt, metadata=metadata)
    except TypeError:
        # Older/newer versions may not support metadata in constructor.
        episode = Episode(prompt=prompt)
        try:
            setattr(episode, "metadata", metadata)
        except Exception:
            pass
        return episode


@dataclass
class TinyCodeVerifiersEnv(SingleTurnEnv):  # type: ignore[misc]
    """Verifiers-compatible wrapper around TinyCodeTestEnvironment."""

    dataset_path: str | Path
    sandbox: SandboxConfig = SandboxConfig()
    strategy: PromptStrategy = PromptStrategy.DIRECT
    difficulty: str | None = None
    adversarial_only: bool = False
    sample_seed: int = 0

    def __post_init__(self) -> None:
        if not VERIFIERS_AVAILABLE:
            raise ImportError(
                "Verifiers is not installed. Install with `pip install verifiers` to use this adapter."
            )
        self._env = TinyCodeTestEnvironment(dataset_path=self.dataset_path, sandbox=self.sandbox)
        self._current_task = None
        self._episodes = 0

    def reset(self) -> Any:
        tasks = self._env.sample(
            n=1,
            difficulty=self.difficulty,
            adversarial_only=self.adversarial_only,
            seed=self.sample_seed + self._episodes,
        )
        if not tasks:
            raise RuntimeError("No tasks available for reset with the current filters.")
        task = tasks[0]
        self._current_task = task
        self._episodes += 1
        return _build_episode(
            prompt=self._env.prompt(task, strategy=self.strategy),
            metadata={
                "task_id": task.task_id,
                "difficulty": task.difficulty,
                "adversarial": task.adversarial,
            },
        )

    def step(self, completion: str) -> float:
        if self._current_task is None:
            raise RuntimeError("Call reset() before step().")
        result = self._env.verify(self._current_task, completion)
        return float(result.reward)


def create_verifiers_env(
    dataset_path: str | Path,
    *,
    sandbox: SandboxConfig | None = None,
    strategy: PromptStrategy = PromptStrategy.DIRECT,
    difficulty: str | None = None,
    adversarial_only: bool = False,
    sample_seed: int = 0,
) -> TinyCodeVerifiersEnv:
    """Factory with a clearer error path when Verifiers is unavailable."""
    if not VERIFIERS_AVAILABLE:
        raise ImportError("Verifiers not available. Install with `pip install verifiers`.")
    return TinyCodeVerifiersEnv(
        dataset_path=dataset_path,
        sandbox=sandbox or SandboxConfig(),
        strategy=strategy,
        difficulty=difficulty,
        adversarial_only=adversarial_only,
        sample_seed=sample_seed,
    )

