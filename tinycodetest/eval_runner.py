from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tinycodetest.harness import HarnessRequest, PromptHarness, PromptStrategy, SandboxConfig
from tinycodetest.models import ModelAdapter
from tinycodetest.schema import Task
from tinycodetest.verifier import verify_completion


@dataclass(frozen=True)
class EvalConfig:
    samples_per_task: int = 5
    ks: tuple[int, ...] = (1, 5)
    seed: int = 0
    capture_attempts: bool = False


def pass_at_k(n: int, c: int, k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    if n < k:
        raise ValueError(f"n ({n}) must be >= k ({k})")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0

    log_prob_no_success = 0.0
    for idx in range(k):
        numerator = n - c - idx
        denominator = n - idx
        log_prob_no_success += math.log(numerator / denominator)
    return 1.0 - math.exp(log_prob_no_success)


def _difficulty_metrics(difficulty: str, rows: list[dict[str, object]], ks: tuple[int, ...]) -> dict[str, float]:
    subset = [row for row in rows if row["difficulty"] == difficulty]
    if not subset:
        return {f"pass@{k}": 0.0 for k in ks}
    out = {}
    for k in ks:
        out[f"pass@{k}"] = sum(float(row[f"pass@{k}"]) for row in subset) / len(subset)
    return out


def evaluate_model(
    *,
    model: ModelAdapter,
    strategy: PromptStrategy,
    tasks: list[Task],
    harness: PromptHarness,
    sandbox: SandboxConfig,
    config: EvalConfig,
) -> dict[str, object]:
    ks = tuple(sorted(set(config.ks)))
    n = max(config.samples_per_task, max(ks))
    rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []

    for idx, task in enumerate(tasks):
        prompt = harness.build_prompt(HarnessRequest(task=task, strategy=strategy))
        completions = model.generate(prompt=prompt, task=task, k=n, seed=config.seed + idx)

        if len(completions) < n:
            completions = completions + [completions[-1]] * (n - len(completions))

        verification = [verify_completion(task, completion, sandbox=sandbox) for completion in completions[:n]]
        successes = sum(1 for result in verification if result.passed)
        mean_reward = sum(result.reward for result in verification) / n
        runtime_ms = sum(result.runtime_ms for result in verification) / n

        row: dict[str, object] = {
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "adversarial": task.adversarial,
            "mean_reward": mean_reward,
            "runtime_ms": runtime_ms,
            "successes": successes,
            "n": n,
        }
        for k in ks:
            row[f"pass@{k}"] = pass_at_k(n=n, c=successes, k=k)
        rows.append(row)

        if config.capture_attempts:
            attempts.append(
                {
                    "task_id": task.task_id,
                    "difficulty": task.difficulty,
                    "adversarial": task.adversarial,
                    "attempts": [result.to_dict() for result in verification],
                }
            )

    total = len(rows)
    adversarial_rows = [row for row in rows if bool(row["adversarial"])]
    overall = {f"pass@{k}": sum(float(row[f"pass@{k}"]) for row in rows) / total for k in ks}
    adv = {
        f"pass@{k}": (
            sum(float(row[f"pass@{k}"]) for row in adversarial_rows) / len(adversarial_rows)
            if adversarial_rows
            else 0.0
        )
        for k in ks
    }

    output: dict[str, object] = {
        "model": model.name,
        "strategy": strategy.value,
        "num_tasks": total,
        "num_adversarial": len(adversarial_rows),
        "overall": {
            **overall,
            "mean_reward": sum(float(row["mean_reward"]) for row in rows) / total,
            "avg_runtime_ms": sum(float(row["runtime_ms"]) for row in rows) / total,
        },
        "adversarial": adv,
        "difficulty": {
            "easy": _difficulty_metrics("easy", rows, ks),
            "medium": _difficulty_metrics("medium", rows, ks),
            "hard": _difficulty_metrics("hard", rows, ks),
        },
        "rows": rows,
    }

    if config.capture_attempts:
        output["attempts"] = attempts

    return output


def evaluate_suite(
    *,
    models: list[ModelAdapter],
    strategies: list[PromptStrategy],
    tasks: list[Task],
    sandbox: SandboxConfig,
    config: EvalConfig,
) -> dict[str, object]:
    harness = PromptHarness(sandbox=sandbox)
    runs: list[dict[str, object]] = []

    for model in models:
        for strategy in strategies:
            runs.append(
                evaluate_model(
                    model=model,
                    strategy=strategy,
                    tasks=tasks,
                    harness=harness,
                    sandbox=sandbox,
                    config=config,
                )
            )

    now = datetime.now(timezone.utc).isoformat()
    return {
        "timestamp_utc": now,
        "config": {
            "samples_per_task": config.samples_per_task,
            "ks": list(config.ks),
            "seed": config.seed,
            "capture_attempts": config.capture_attempts,
        },
        "runs": runs,
    }


def leaderboard_markdown(payload: dict[str, object]) -> str:
    runs = list(payload["runs"])
    config = dict(payload.get("config", {}))
    ks = sorted(int(k) for k in config.get("ks", [1, 5]))
    pass_columns = [f"pass@{k}" for k in ks]
    adv_focus = "pass@1" if 1 in ks else pass_columns[0]

    def sort_key(item: dict[str, object]) -> tuple[float, float]:
        overall = item["overall"]
        primary = float(overall.get(pass_columns[0], 0.0))
        secondary = float(overall.get(pass_columns[-1], 0.0))
        return (primary, secondary)

    runs.sort(key=sort_key, reverse=True)

    pass_headers = " | ".join(pass_columns)
    pass_align = " | ".join(["---:"] * len(pass_columns))
    lines = [
        "# TinyCodeTest Leaderboard",
        "",
        f"Generated (UTC): {payload['timestamp_utc']}",
        "",
        f"| Rank | Model | Strategy | {pass_headers} | Mean Reward | Adv {adv_focus} |",
        f"| --- | --- | --- | {pass_align} | ---: | ---: |",
    ]

    for rank, run in enumerate(runs, start=1):
        overall = run["overall"]
        adversarial = run["adversarial"]
        pass_values = " | ".join(f"{float(overall.get(column, 0.0)):.3f}" for column in pass_columns)
        lines.append(
            "| "
            f"{rank} | {run['model']} | {run['strategy']} | "
            f"{pass_values} | "
            f"{float(overall.get('mean_reward', 0.0)):.3f} | "
            f"{float(adversarial.get(adv_focus, 0.0)):.3f} |"
        )

    return "\n".join(lines) + "\n"


def save_eval(payload: dict[str, object], output_dir: str | Path, stem: str) -> tuple[Path, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    json_path = target_dir / f"{stem}.json"
    md_path = target_dir / f"{stem}.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(leaderboard_markdown(payload), encoding="utf-8")

    latest_json = target_dir / "latest.json"
    latest_md = target_dir / "latest.md"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    return json_path, md_path
