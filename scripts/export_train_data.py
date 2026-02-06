#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinycodetest.dataset import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TinyCodeTest dataset into trainable prompt/completion JSONL.")
    parser.add_argument("--dataset", required=True, help="Input dataset JSONL")
    parser.add_argument("--output", required=True, help="Output training JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_dataset(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            row = {
                "task_id": task.task_id,
                "difficulty": task.difficulty,
                "adversarial": task.adversarial,
                "prompt": task.prompt,
                "completion": task.canonical_solution,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a coding model. Return only Python function code.",
                    },
                    {"role": "user", "content": task.prompt},
                ],
                "metadata": task.metadata or {},
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    print(f"Exported {len(tasks)} records to {output_path}")


if __name__ == "__main__":
    main()
