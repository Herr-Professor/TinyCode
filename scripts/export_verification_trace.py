#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinycodetest.eval_runner import save_verification_trace
from tinycodetest.reporting import load_eval_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TinyCodeTest verification trace markdown.")
    parser.add_argument("--input", required=True, help="Path to eval JSON payload")
    parser.add_argument("--output", required=True, help="Path to output markdown file")
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=50,
        help="Maximum tasks per model/strategy section",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_eval_payload(args.input)
    path = save_verification_trace(payload, args.output, max_tasks=max(1, args.max_tasks))
    print(f"Verification trace: {path}")


if __name__ == "__main__":
    main()
