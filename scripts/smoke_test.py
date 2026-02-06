#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def _ks_for_samples(samples_per_task: int) -> str:
    if samples_per_task >= 5:
        return "1,5"
    if samples_per_task >= 3:
        return "1,3"
    if samples_per_task >= 2:
        return "1,2"
    return "1"


def _run_smoke(
    *,
    output_root: Path,
    python_bin: str,
    tasks: int,
    seed: int,
    samples_per_task: int,
    max_tasks: int,
) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = output_root / "smoke.jsonl"
    adversarial_path = output_root / "smoke_adversarial.jsonl"
    result_dir = output_root / "smoke_results"
    stem = "smoke"

    _run(
        [
            python_bin,
            "scripts/generate_dataset.py",
            "--output",
            str(dataset_path),
            "--adversarial-output",
            str(adversarial_path),
            "--tasks",
            str(tasks),
            "--seed",
            str(seed),
        ]
    )

    _run(
        [
            python_bin,
            "scripts/run_eval.py",
            "--dataset",
            str(dataset_path),
            "--models",
            "heuristic-small,reference-oracle",
            "--strategies",
            "direct",
            "--samples-per-task",
            str(samples_per_task),
            "--ks",
            _ks_for_samples(samples_per_task),
            "--max-tasks",
            str(max_tasks),
            "--seed",
            str(seed),
            "--confidence-intervals",
            "--capture-attempts",
            "--export-verification-trace",
            "--generate-report",
            "--output-dir",
            str(result_dir),
            "--stem",
            stem,
        ]
    )

    expected = [
        result_dir / f"{stem}.json",
        result_dir / f"{stem}.md",
        result_dir / f"{stem}.html",
        result_dir / f"{stem}.verification.md",
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise RuntimeError(f"Smoke test missing expected artifacts: {joined}")

    payload = json.loads((result_dir / f"{stem}.json").read_text(encoding="utf-8"))
    if not bool(dict(payload.get("config", {})).get("confidence_intervals", False)):
        raise RuntimeError("Smoke test expected confidence_intervals=true in payload config.")

    return expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TinyCodeTest smoke test (dataset + eval + report + verification trace)."
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="Directory for smoke artifacts. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep artifacts under results/smoke_test when --output-root is not set.",
    )
    parser.add_argument("--tasks", type=int, default=12, help="Task count for generated smoke dataset")
    parser.add_argument("--samples-per-task", type=int, default=3, help="Completions per task")
    parser.add_argument("--max-tasks", type=int, default=12, help="Max tasks for eval")
    parser.add_argument("--seed", type=int, default=42, help="Seed for generation and eval")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for subprocess commands")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = max(1, args.tasks)
    samples_per_task = max(1, args.samples_per_task)
    max_tasks = max(1, args.max_tasks)

    if args.output_root:
        root = Path(args.output_root)
        if not root.is_absolute():
            root = (ROOT / root).resolve()
        artifacts = _run_smoke(
            output_root=root,
            python_bin=args.python_bin,
            tasks=tasks,
            seed=args.seed,
            samples_per_task=samples_per_task,
            max_tasks=max_tasks,
        )
        print(f"Smoke test passed. Artifacts in {root}")
        for artifact in artifacts:
            print(f"- {artifact}")
        return

    if args.keep_artifacts:
        root = ROOT / "results" / "smoke_test"
        artifacts = _run_smoke(
            output_root=root,
            python_bin=args.python_bin,
            tasks=tasks,
            seed=args.seed,
            samples_per_task=samples_per_task,
            max_tasks=max_tasks,
        )
        print(f"Smoke test passed. Artifacts in {root}")
        for artifact in artifacts:
            print(f"- {artifact}")
        return

    with tempfile.TemporaryDirectory(prefix="tinycodetest_smoke_") as temp_dir:
        root = Path(temp_dir)
        artifacts = _run_smoke(
            output_root=root,
            python_bin=args.python_bin,
            tasks=tasks,
            seed=args.seed,
            samples_per_task=samples_per_task,
            max_tasks=max_tasks,
        )
        print(f"Smoke test passed (temporary run). Artifact root: {root}")
        print("Temporary artifacts are deleted when this command exits. Use --keep-artifacts to persist.")
        for artifact in artifacts:
            print(f"- {artifact}")


if __name__ == "__main__":
    main()
