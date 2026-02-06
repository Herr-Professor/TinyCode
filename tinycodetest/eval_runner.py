from __future__ import annotations

import json
import math
import random
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
    confidence_intervals: bool = False
    confidence_level: float = 0.95
    bootstrap_samples: int = 2000


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


def _stable_text_seed(text: str) -> int:
    seed = 0
    for char in text:
        seed = ((seed * 131) + ord(char)) & 0x7FFFFFFF
    return seed


def _percentile_sorted(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0.0:
        return values[0]
    if q >= 1.0:
        return values[-1]

    position = q * (len(values) - 1)
    low_idx = int(math.floor(position))
    high_idx = int(math.ceil(position))
    if low_idx == high_idx:
        return values[low_idx]

    fraction = position - low_idx
    low = values[low_idx]
    high = values[high_idx]
    return low + ((high - low) * fraction)


def _bootstrap_mean_interval(
    values: list[float],
    *,
    confidence: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)

    mean = sum(values) / len(values)
    if len(values) == 1:
        return (mean, mean, mean)

    sample_count = max(200, bootstrap_samples)
    rng = random.Random(seed)
    n = len(values)
    estimates: list[float] = []
    for _ in range(sample_count):
        sample_sum = 0.0
        for _ in range(n):
            sample_sum += values[rng.randrange(n)]
        estimates.append(sample_sum / n)

    estimates.sort()
    alpha = max(0.0001, min(0.99, 1.0 - confidence))
    lower = _percentile_sorted(estimates, alpha / 2.0)
    upper = _percentile_sorted(estimates, 1.0 - (alpha / 2.0))
    return (mean, lower, upper)


def _metric_confidence_intervals(
    rows: list[dict[str, object]],
    ks: tuple[int, ...],
    *,
    confidence: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for idx, k in enumerate(ks):
        metric = f"pass@{k}"
        values = [float(row.get(metric, 0.0)) for row in rows]
        estimate, lower, upper = _bootstrap_mean_interval(
            values,
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            seed=seed + ((idx + 1) * 1543),
        )
        out[metric] = [estimate, lower, upper]
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
    confidence = max(0.01, min(0.999, config.confidence_level))
    run_seed = (config.seed * 1000003) + _stable_text_seed(f"{model.name}|{strategy.value}")
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

    if config.confidence_intervals:
        output["confidence_intervals"] = {
            "confidence_level": confidence,
            "method": "bootstrap_mean",
            "bootstrap_samples": max(200, config.bootstrap_samples),
            "overall": _metric_confidence_intervals(
                rows,
                ks,
                confidence=confidence,
                bootstrap_samples=config.bootstrap_samples,
                seed=run_seed + 101,
            ),
            "adversarial": _metric_confidence_intervals(
                adversarial_rows,
                ks,
                confidence=confidence,
                bootstrap_samples=config.bootstrap_samples,
                seed=run_seed + 211,
            ),
            "difficulty": {
                "easy": _metric_confidence_intervals(
                    [row for row in rows if row["difficulty"] == "easy"],
                    ks,
                    confidence=confidence,
                    bootstrap_samples=config.bootstrap_samples,
                    seed=run_seed + 307,
                ),
                "medium": _metric_confidence_intervals(
                    [row for row in rows if row["difficulty"] == "medium"],
                    ks,
                    confidence=confidence,
                    bootstrap_samples=config.bootstrap_samples,
                    seed=run_seed + 401,
                ),
                "hard": _metric_confidence_intervals(
                    [row for row in rows if row["difficulty"] == "hard"],
                    ks,
                    confidence=confidence,
                    bootstrap_samples=config.bootstrap_samples,
                    seed=run_seed + 503,
                ),
            },
        }

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
    config_payload: dict[str, object] = {
        "samples_per_task": config.samples_per_task,
        "ks": list(config.ks),
        "seed": config.seed,
        "capture_attempts": config.capture_attempts,
    }
    if config.confidence_intervals:
        confidence = max(0.01, min(0.999, config.confidence_level))
        config_payload["confidence_intervals"] = True
        config_payload["confidence_level"] = confidence
        config_payload["bootstrap_samples"] = max(200, config.bootstrap_samples)

    return {
        "timestamp_utc": now,
        "config": config_payload,
        "runs": runs,
    }


def leaderboard_markdown(payload: dict[str, object]) -> str:
    runs = list(payload["runs"])
    config = dict(payload.get("config", {}))
    ks = sorted(int(k) for k in config.get("ks", [1, 5]))
    pass_columns = [f"pass@{k}" for k in ks]
    adv_focus = "pass@1" if 1 in ks else pass_columns[0]
    show_ci = bool(config.get("confidence_intervals", False))
    try:
        confidence_level = float(config.get("confidence_level", 0.95) or 0.95)
    except Exception:
        confidence_level = 0.95
    confidence_pct = int(round(confidence_level * 100.0))

    def ci_text(run: dict[str, object], metric: str) -> str:
        ci_root = dict(run.get("confidence_intervals", {}))
        overall_ci = dict(ci_root.get("overall", {}))
        raw = overall_ci.get(metric)
        if not isinstance(raw, list) or len(raw) != 3:
            return "-"
        low = float(raw[1])
        high = float(raw[2])
        return f"[{low:.3f}, {high:.3f}]"

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
    ]
    if show_ci:
        lines.append(
            f"| Rank | Model | Strategy | {pass_headers} | {confidence_pct}% CI ({pass_columns[0]}) | Mean Reward | Adv {adv_focus} |"
        )
        lines.append(f"| --- | --- | --- | {pass_align} | :---: | ---: | ---: |")
    else:
        lines.append(f"| Rank | Model | Strategy | {pass_headers} | Mean Reward | Adv {adv_focus} |")
        lines.append(f"| --- | --- | --- | {pass_align} | ---: | ---: |")

    for rank, run in enumerate(runs, start=1):
        overall = run["overall"]
        adversarial = run["adversarial"]
        pass_values = " | ".join(f"{float(overall.get(column, 0.0)):.3f}" for column in pass_columns)
        if show_ci:
            lines.append(
                "| "
                f"{rank} | {run['model']} | {run['strategy']} | "
                f"{pass_values} | "
                f"{ci_text(run, pass_columns[0])} | "
                f"{float(overall.get('mean_reward', 0.0)):.3f} | "
                f"{float(adversarial.get(adv_focus, 0.0)):.3f} |"
            )
        else:
            lines.append(
                "| "
                f"{rank} | {run['model']} | {run['strategy']} | "
                f"{pass_values} | "
                f"{float(overall.get('mean_reward', 0.0)):.3f} | "
                f"{float(adversarial.get(adv_focus, 0.0)):.3f} |"
            )

    return "\n".join(lines) + "\n"


def verification_trace_markdown(payload: dict[str, object], *, max_tasks: int = 50) -> str:
    lines: list[str] = [
        "# Verification Trace",
        "",
        "This trace shows verifier outcomes for sampled attempts per task.",
        "",
    ]

    runs = list(payload.get("runs", []))
    if not runs:
        lines.append("_No runs found in payload._")
        return "\n".join(lines) + "\n"

    for run in runs:
        model = run.get("model", "")
        strategy = run.get("strategy", "")
        lines.append(f"## {model} / {strategy}")
        attempts = list(run.get("attempts", []))
        if not attempts:
            lines.append("- capture_attempts disabled for this run.")
            lines.append("")
            continue

        for record in attempts[:max_tasks]:
            task_id = record.get("task_id", "")
            task_attempts = list(record.get("attempts", []))
            lines.append(f"- **{task_id}**")
            for idx, attempt in enumerate(task_attempts, start=1):
                passed = bool(attempt.get("passed", False))
                verdict = "PASS" if passed else "FAIL"
                passed_cases = int(attempt.get("passed_cases", 0))
                total_cases = int(attempt.get("total_cases", 0))
                reward = float(attempt.get("reward", 0.0))
                error = attempt.get("error")
                msg = (
                    f"  - attempt {idx}: {verdict}, reward={reward:.3f}, "
                    f"cases={passed_cases}/{total_cases}"
                )
                if error:
                    msg += f", error={error}"
                lines.append(msg)

                failures = list(attempt.get("failures", []))
                if failures:
                    first = failures[0]
                    reason = first.get("reason", "")
                    case_idx = first.get("idx", "")
                    lines.append(f"    - first failure: case={case_idx}, reason={reason}")
            lines.append("")

    return "\n".join(lines) + "\n"


def save_verification_trace(
    payload: dict[str, object],
    output_path: str | Path,
    *,
    max_tasks: int = 50,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(verification_trace_markdown(payload, max_tasks=max_tasks), encoding="utf-8")
    return path


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
