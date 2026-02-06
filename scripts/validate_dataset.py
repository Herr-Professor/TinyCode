#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinycodetest.dataset import dataset_summary, load_dataset
from tinycodetest.harness import SandboxConfig
from tinycodetest.verifier import verify_completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a TinyCodeTest-format dataset.")
    parser.add_argument("--dataset", required=True, help="Dataset JSONL path")
    parser.add_argument("--check-canonical", action="store_true", help="Run canonical solutions through verifier")
    parser.add_argument("--timeout", type=float, default=1.0, help="Timeout for canonical verification")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_dataset(args.dataset)
    summary = dataset_summary(tasks)

    print("Dataset loaded")
    print(f"- total: {summary.total_tasks}")
    print(f"- easy: {summary.easy}")
    print(f"- medium: {summary.medium}")
    print(f"- hard: {summary.hard}")
    print(f"- adversarial: {summary.adversarial}")

    if args.check_canonical:
        sandbox = SandboxConfig(timeout_seconds=args.timeout, memory_mb=256)
        failed = []
        for task in tasks:
            result = verify_completion(task, task.canonical_solution, sandbox=sandbox)
            if not result.passed:
                failed.append((task.task_id, result.error))
        print(f"- canonical_failed: {len(failed)}")
        if failed:
            print("Failed task IDs:")
            for task_id, error in failed[:30]:
                print(f"  - {task_id}: {error}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
