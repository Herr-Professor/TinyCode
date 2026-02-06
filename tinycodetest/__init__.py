from tinycodetest.dataset import (
    EASY,
    HARD,
    MEDIUM,
    TaskSampler,
    build_dataset,
    dataset_summary,
    generate_and_write,
    load_dataset,
    validate_dataset_uniqueness,
    write_dataset,
)
from tinycodetest.environment import TinyCodeTestEnvironment
from tinycodetest.harness import PromptHarness, PromptStrategy, SandboxConfig
from tinycodetest.schema import Task, TestCase, VerificationResult
from tinycodetest.verifier import verify_completion

try:
    from tinycodetest.verifiers_adapter import VERIFIERS_AVAILABLE, TinyCodeVerifiersEnv, create_verifiers_env
except Exception:  # pragma: no cover - optional dependency surface
    VERIFIERS_AVAILABLE = False
    TinyCodeVerifiersEnv = None  # type: ignore[assignment]
    create_verifiers_env = None  # type: ignore[assignment]

__all__ = [
    "EASY",
    "MEDIUM",
    "HARD",
    "Task",
    "TestCase",
    "TaskSampler",
    "VerificationResult",
    "PromptHarness",
    "PromptStrategy",
    "SandboxConfig",
    "TinyCodeTestEnvironment",
    "build_dataset",
    "write_dataset",
    "load_dataset",
    "dataset_summary",
    "generate_and_write",
    "validate_dataset_uniqueness",
    "verify_completion",
    "VERIFIERS_AVAILABLE",
    "TinyCodeVerifiersEnv",
    "create_verifiers_env",
]
