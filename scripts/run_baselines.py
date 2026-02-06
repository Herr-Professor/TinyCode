#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run default TinyCodeTest baseline eval suite.")
    parser.add_argument("--dataset", default="data/tinycodetest.jsonl")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-tasks", type=int, default=150)
    parser.add_argument("--samples-per-task", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--include-oracle", action="store_true", help="Include reference-oracle baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = "heuristic-small,heuristic-medium"
    if args.include_oracle:
        models += ",reference-oracle"

    command = [
        sys.executable,
        "scripts/run_eval.py",
        "--dataset",
        args.dataset,
        "--auto-generate",
        "--models",
        models,
        "--strategies",
        "direct,plan_then_code",
        "--ks",
        "1,5",
        "--samples-per-task",
        str(args.samples_per_task),
        "--max-tasks",
        str(args.max_tasks),
        "--seed",
        str(args.seed),
        "--output-dir",
        args.output_dir,
    ]

    completed = subprocess.run(command, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
