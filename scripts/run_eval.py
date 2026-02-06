#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinycodetest.dataset import TaskSampler, dataset_summary, generate_and_write, load_dataset
from tinycodetest.eval_runner import (
    EvalConfig,
    evaluate_suite,
    leaderboard_markdown,
    save_eval,
    save_verification_trace,
)
from tinycodetest.harness import PromptStrategy, SandboxConfig
from tinycodetest.models import resolve_models
from tinycodetest.reporting import save_html_report


def _parse_csv(values: str) -> list[str]:
    return [value.strip() for value in values.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    default_sandbox = SandboxConfig.for_environment(serverless=False)
    parser = argparse.ArgumentParser(description="Run TinyCodeTest benchmark evaluation.")
    parser.add_argument("--dataset", default="data/tinycodetest.jsonl", help="Dataset JSONL path")
    parser.add_argument("--auto-generate", action="store_true", help="Generate dataset if missing")
    parser.add_argument("--tasks", type=int, default=600, help="Dataset size when auto-generating")
    parser.add_argument("--seed", type=int, default=13, help="Evaluation seed")
    parser.add_argument("--dataset-seed", type=int, default=7, help="Generation seed for auto-generate")
    parser.add_argument("--models", default="heuristic-small,heuristic-medium", help="Comma-separated model names")
    parser.add_argument(
        "--model-config",
        default="",
        help="Optional JSON config path for custom model registry (multi-endpoint support)",
    )
    parser.add_argument("--strategies", default="direct,plan_then_code", help="Comma-separated prompt strategies")
    parser.add_argument("--samples-per-task", type=int, default=5, help="Number of completions per task")
    parser.add_argument("--ks", default="1,5", help="Comma-separated pass@k values")
    parser.add_argument("--max-tasks", type=int, default=0, help="Evaluate only a random subset (0 = all)")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--adversarial-only", action="store_true", help="Use only adversarial tasks")
    parser.add_argument(
        "--timeout",
        type=float,
        default=default_sandbox.timeout_seconds,
        help="Verifier timeout per attempt (seconds)",
    )
    parser.add_argument(
        "--memory-mb",
        type=int,
        default=default_sandbox.memory_mb,
        help="Memory cap for verifier subprocess",
    )
    parser.add_argument("--output-dir", default="results", help="Directory for JSON + leaderboard outputs")
    parser.add_argument("--capture-attempts", action="store_true", help="Include per-attempt records in JSON")
    parser.add_argument(
        "--confidence-intervals",
        action="store_true",
        help="Include bootstrap 95%% confidence intervals for pass@k in JSON/Markdown/HTML outputs",
    )
    parser.add_argument(
        "--export-verification-trace",
        action="store_true",
        help="Write a human-readable verification trace markdown file (requires --capture-attempts for rich output)",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate an HTML dashboard report next to eval outputs",
    )
    parser.add_argument("--stem", default="", help="Optional filename stem")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        if not args.auto_generate:
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}. Use --auto-generate or run scripts/generate_dataset.py first."
            )
        summary = generate_and_write(
            output_path=dataset_path,
            adversarial_output_path=dataset_path.with_name("tinycodetest_adversarial.jsonl"),
            total_tasks=args.tasks,
            seed=args.dataset_seed,
        )
        print(
            "Auto-generated dataset "
            f"({summary.total_tasks} tasks; adv={summary.adversarial}) at {dataset_path}"
        )

    tasks = load_dataset(dataset_path)
    sampler = TaskSampler(tasks)

    if args.difficulty or args.adversarial_only:
        filtered = sampler.sample(
            n=len(tasks),
            difficulty=args.difficulty,
            adversarial_only=args.adversarial_only,
            seed=args.seed,
        )
        tasks = filtered

    if args.max_tasks > 0 and args.max_tasks < len(tasks):
        tasks = TaskSampler(tasks).sample(n=args.max_tasks, seed=args.seed)

    if not tasks:
        raise RuntimeError("No tasks selected for evaluation.")

    model_names = _parse_csv(args.models)
    strategy_names = _parse_csv(args.strategies)
    strategies = [PromptStrategy(name) for name in strategy_names]
    ks = tuple(int(value) for value in _parse_csv(args.ks))

    model_config_path = args.model_config or None
    models = resolve_models(model_names, model_config_path=model_config_path)
    sandbox = SandboxConfig(timeout_seconds=args.timeout, memory_mb=args.memory_mb)
    config = EvalConfig(
        samples_per_task=args.samples_per_task,
        ks=ks,
        seed=args.seed,
        capture_attempts=args.capture_attempts,
        confidence_intervals=args.confidence_intervals,
    )

    payload = evaluate_suite(
        models=models,
        strategies=strategies,
        tasks=tasks,
        sandbox=sandbox,
        config=config,
    )
    payload["config"]["model_config"] = str(model_config_path) if model_config_path else None
    payload["dataset"] = {
        "path": str(dataset_path),
        "selected_tasks": len(tasks),
        "summary": dataset_summary(tasks).__dict__,
        "difficulty_filter": args.difficulty,
        "adversarial_only": args.adversarial_only,
    }

    if args.stem:
        stem = args.stem
    else:
        stamp = datetime.now(timezone.utc).strftime("eval_%Y%m%d_%H%M%S")
        stem = stamp

    json_path, md_path = save_eval(payload, output_dir=args.output_dir, stem=stem)
    html_path = None
    trace_path = None
    if args.generate_report:
        html_path = save_html_report(payload, Path(args.output_dir) / f"{stem}.html")
    if args.export_verification_trace:
        trace_path = save_verification_trace(payload, Path(args.output_dir) / f"{stem}.verification.md")

    print(leaderboard_markdown(payload))
    print(f"JSON: {json_path}")
    print(f"Leaderboard: {md_path}")
    if html_path is not None:
        print(f"HTML report: {html_path}")
    if trace_path is not None:
        print(f"Verification trace: {trace_path}")


if __name__ == "__main__":
    main()
