#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinycodetest.reporting import load_eval_payload, save_html_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HTML report from TinyCodeTest eval JSON.")
    parser.add_argument("--input", required=True, help="Eval JSON path (from run_eval.py)")
    parser.add_argument("--output", default="", help="HTML output path (default: same stem as input)")
    parser.add_argument("--title", default="TinyCodeTest Report", help="Report title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    payload = load_eval_payload(input_path)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".html")

    saved = save_html_report(payload, output_path, title=args.title)
    print(f"HTML report: {saved}")


if __name__ == "__main__":
    main()
