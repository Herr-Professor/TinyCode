#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinycodetest.dataset import generate_and_write


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the TinyCodeTest dataset.")
    parser.add_argument("--output", default="data/tinycodetest.jsonl", help="Path to dataset JSONL file")
    parser.add_argument(
        "--adversarial-output",
        default="data/tinycodetest_adversarial.jsonl",
        help="Path to adversarial-only JSONL file",
    )
    parser.add_argument("--tasks", type=int, default=600, help="Total number of tasks")
    parser.add_argument("--seed", type=int, default=7, help="Dataset generation seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_and_write(
        output_path=args.output,
        adversarial_output_path=args.adversarial_output,
        total_tasks=args.tasks,
        seed=args.seed,
    )
    print("Generated TinyCodeTest dataset")
    print(f"- total: {summary.total_tasks}")
    print(f"- easy: {summary.easy}")
    print(f"- medium: {summary.medium}")
    print(f"- hard: {summary.hard}")
    print(f"- adversarial: {summary.adversarial}")
    print(f"- output: {args.output}")
    print(f"- adversarial output: {args.adversarial_output}")


if __name__ == "__main__":
    main()
