from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _esc(value: object) -> str:
    return html.escape(str(value))


def _compact_text(value: object, *, max_len: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return "-"
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _sort_runs(payload: dict[str, object]) -> list[dict[str, object]]:
    config = dict(payload.get("config", {}))
    ks = sorted(_to_int(value, 1) for value in config.get("ks", [1]))
    primary = f"pass@{ks[0]}"
    secondary = f"pass@{ks[-1]}"
    runs = list(payload.get("runs", []))
    runs.sort(
        key=lambda run: (
            _to_float(dict(run.get("overall", {})).get(primary, 0.0)),
            _to_float(dict(run.get("overall", {})).get(secondary, 0.0)),
        ),
        reverse=True,
    )
    return runs


def _polyline(values: list[float], width: int, height: int, pad: int = 16) -> str:
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    if hi == lo:
        hi = lo + 1e-9

    span_x = max(1, len(values) - 1)
    points = []
    for idx, value in enumerate(values):
        x = pad + (idx / span_x) * (width - (2 * pad))
        y_norm = (value - lo) / (hi - lo)
        y = (height - pad) - y_norm * (height - (2 * pad))
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _histogram(values: list[float], bins: int = 10) -> tuple[list[str], list[int]]:
    if not values:
        return [], []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        hi = lo + 1e-9

    width = (hi - lo) / bins
    counts = [0 for _ in range(bins)]
    for value in values:
        idx = int((value - lo) / width)
        if idx == bins:
            idx -= 1
        counts[idx] += 1

    labels = []
    for idx in range(bins):
        start = lo + (idx * width)
        end = start + width
        labels.append(f"{start:.3f}-{end:.3f}")
    return labels, counts


def _bar_chart(
    labels: list[str],
    values: list[float],
    width: int,
    height: int,
    *,
    bar_color: str,
    label_color: str,
    grid_color: str,
    pad: int = 24,
) -> str:
    if not values:
        return ""
    top = max(values)
    if top <= 0:
        top = 1e-9

    n = len(values)
    gap = 6
    usable = width - (2 * pad)
    bar_w = max(3, int((usable - gap * (n - 1)) / max(1, n)))

    parts: list[str] = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        y = height - pad - frac * (height - (2 * pad))
        parts.append(
            f'<line x1="{pad}" y1="{y:.1f}" x2="{width - pad}" y2="{y:.1f}" stroke="{grid_color}" stroke-width="1" />'
        )

    for idx, value in enumerate(values):
        x = pad + idx * (bar_w + gap)
        bar_h = ((value / top) * (height - (2 * pad))) if value > 0 else 0.0
        y = height - pad - bar_h
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="4" fill="{bar_color}" />'
        )

        label = _esc(labels[idx][:18])
        parts.append(
            f'<text x="{(x + bar_w / 2):.1f}" y="{height - 6}" fill="{label_color}" font-size="10" text-anchor="middle">{label}</text>'
        )
    return "".join(parts)


def _error_kind(attempt: dict[str, object]) -> str:
    if bool(attempt.get("passed", False)):
        return "passed"

    if bool(attempt.get("timed_out", False)):
        return "timeout"

    error = str(attempt.get("error") or "")
    if error.startswith("compile_error"):
        return "compile_error"
    if error.startswith("missing_function"):
        return "missing_function"
    if error.startswith("timeout"):
        return "timeout"

    failures = list(attempt.get("failures", []))
    reason = str(failures[0].get("reason", "")) if failures else ""
    if reason.startswith("runtime_error"):
        return "runtime_error"
    if "assertion_failed" in reason:
        return "assertion_failed"

    if error:
        return error.split(":", 1)[0]
    return "verification_failed"


def _error_label(kind: str) -> str:
    mapping = {
        "passed": "Passed",
        "assertion_failed": "Assertion failed",
        "runtime_error": "Runtime error",
        "compile_error": "Compile error",
        "missing_function": "Missing function",
        "timeout": "Timeout",
        "verification_failed": "Verification failed",
    }
    return mapping.get(kind, kind.replace("_", " ").title())


def _extract_error_detail(attempt: dict[str, object]) -> str:
    error = str(attempt.get("error") or "")
    failures = list(attempt.get("failures", []))
    first_reason = str(failures[0].get("reason", "")) if failures else ""

    if error and error != "verification_failed":
        # Keep dashboard details concise; traceback text is still in raw JSON if needed.
        return _compact_text(error, max_len=180)
    if first_reason:
        cleaned = first_reason.strip()
        if cleaned.endswith(":"):
            cleaned = cleaned[:-1]
        return _compact_text(cleaned or first_reason, max_len=180)
    if error:
        return _compact_text(error, max_len=180)
    return "-"


def _collect_verification_details(
    runs: list[dict[str, object]],
) -> tuple[list[dict[str, object]], Counter[str], list[dict[str, object]]]:
    run_summaries: list[dict[str, object]] = []
    failure_counts: Counter[str] = Counter()
    trace_sections: list[dict[str, object]] = []

    for run in runs:
        model = str(run.get("model", ""))
        strategy = str(run.get("strategy", ""))
        key = f"{model}|{strategy}"

        records = list(run.get("attempts", []))
        if not records:
            continue

        run_failure_counts: Counter[str] = Counter()
        task_cards: list[dict[str, object]] = []
        total_attempts = 0
        passed_attempts = 0
        reward_sum = 0.0

        for record in records:
            task_id = str(record.get("task_id", ""))
            difficulty = str(record.get("difficulty", ""))
            attempts = list(record.get("attempts", []))
            if not attempts:
                continue

            attempt_rows: list[dict[str, object]] = []
            task_passed = 0
            task_reward_sum = 0.0
            first_failure = "-"

            for idx, attempt_raw in enumerate(attempts, start=1):
                attempt = dict(attempt_raw)
                passed = bool(attempt.get("passed", False))
                reward = _to_float(attempt.get("reward", 0.0))
                passed_cases = _to_int(attempt.get("passed_cases", 0))
                total_cases = _to_int(attempt.get("total_cases", 0))
                runtime_ms = _to_float(attempt.get("runtime_ms", 0.0))

                total_attempts += 1
                reward_sum += reward
                task_reward_sum += reward

                if passed:
                    passed_attempts += 1
                    task_passed += 1

                kind = _error_kind(attempt)
                detail = _extract_error_detail(attempt)
                if not passed and first_failure == "-":
                    first_failure = detail

                if kind != "passed":
                    run_failure_counts[kind] += 1
                    failure_counts[kind] += 1

                attempt_rows.append(
                    {
                        "idx": idx,
                        "passed": passed,
                        "passed_cases": passed_cases,
                        "total_cases": total_cases,
                        "reward": reward,
                        "runtime_ms": runtime_ms,
                        "detail": detail,
                    }
                )

            task_total = len(attempt_rows)
            task_cards.append(
                {
                    "task_id": task_id,
                    "difficulty": difficulty,
                    "attempt_total": task_total,
                    "attempt_passed": task_passed,
                    "mean_reward": (task_reward_sum / task_total) if task_total else 0.0,
                    "first_failure": first_failure,
                    "attempt_rows": attempt_rows,
                }
            )

        if total_attempts == 0:
            continue

        dominant_failure = run_failure_counts.most_common(1)[0][0] if run_failure_counts else "passed"
        run_summaries.append(
            {
                "key": key,
                "model": model,
                "strategy": strategy,
                "attempt_total": total_attempts,
                "attempt_passed": passed_attempts,
                "attempt_pass_rate": passed_attempts / total_attempts,
                "mean_reward": reward_sum / total_attempts,
                "dominant_failure": dominant_failure,
            }
        )

        trace_sections.append(
            {
                "key": key,
                "model": model,
                "strategy": strategy,
                "task_count": len(task_cards),
                "attempt_total": total_attempts,
                "attempt_passed": passed_attempts,
                "attempt_pass_rate": passed_attempts / total_attempts,
                "tasks": task_cards,
            }
        )

    return run_summaries, failure_counts, trace_sections


def render_report_html(payload: dict[str, object], *, title: str = "TinyCodeTest Report") -> str:
    runs = _sort_runs(payload)
    if not runs:
        raise ValueError("No runs available in payload.")

    best = runs[0]
    best_rows = list(best.get("rows", []))
    reward_series = [_to_float(dict(row).get("mean_reward", 0.0)) for row in best_rows]
    reward_points = _polyline(reward_series, width=760, height=260)

    bins, counts = _histogram(reward_series, bins=10)
    hist_svg = _bar_chart(
        bins,
        [_to_float(x) for x in counts],
        width=760,
        height=240,
        bar_color="#e67e45",
        label_color="#697577",
        grid_color="#eef1e8",
    )

    config = dict(payload.get("config", {}))
    ks = sorted(_to_int(k, 1) for k in config.get("ks", [1]))
    pass_key = f"pass@{ks[0]}"

    model_labels = [f"{run.get('model', '')}/{run.get('strategy', '')}" for run in runs]
    model_scores = [_to_float(dict(run.get("overall", {})).get(pass_key, 0.0)) for run in runs]
    model_svg = _bar_chart(
        model_labels,
        model_scores,
        width=760,
        height=240,
        bar_color="#1f8a8a",
        label_color="#697577",
        grid_color="#eef1e8",
    )

    difficulty = dict(best.get("difficulty", {}))
    diff_labels = ["easy", "medium", "hard"]
    diff_scores = [_to_float(dict(difficulty.get(lbl, {})).get(pass_key, 0.0)) for lbl in diff_labels]
    diff_svg = _bar_chart(
        diff_labels,
        diff_scores,
        width=420,
        height=240,
        bar_color="#2f9f74",
        label_color="#697577",
        grid_color="#eef1e8",
    )

    verify_summaries, failure_counts, trace_sections = _collect_verification_details(runs)
    verify_summary_map = {str(item["key"]): item for item in verify_summaries}

    run_rows = []
    for idx, run in enumerate(runs, start=1):
        overall = dict(run.get("overall", {}))
        adv = dict(run.get("adversarial", {}))
        key = f"{run.get('model', '')}|{run.get('strategy', '')}"
        verify = verify_summary_map.get(key)
        verify_rate = "n/a"
        if verify is not None:
            verify_rate = f"{_to_float(verify.get('attempt_pass_rate', 0.0)) * 100.0:.1f}%"

        run_rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{_esc(run.get('model', ''))}</td>"
            f"<td>{_esc(run.get('strategy', ''))}</td>"
            f"<td>{_to_float(overall.get(pass_key, 0.0)):.3f}</td>"
            f"<td>{_to_float(overall.get(f'pass@{ks[-1]}', 0.0)):.3f}</td>"
            f"<td>{_to_float(adv.get(pass_key, 0.0)):.3f}</td>"
            f"<td>{_to_float(overall.get('mean_reward', 0.0)):.3f}</td>"
            f"<td>{_to_float(overall.get('avg_runtime_ms', 0.0)):.1f}</td>"
            f"<td>{verify_rate}</td>"
            "</tr>"
        )

    verify_rows = []
    for item in verify_summaries:
        verify_rows.append(
            "<tr>"
            f"<td>{_esc(item.get('model', ''))}</td>"
            f"<td>{_esc(item.get('strategy', ''))}</td>"
            f"<td>{_to_int(item.get('attempt_total', 0))}</td>"
            f"<td>{_to_int(item.get('attempt_passed', 0))}</td>"
            f"<td>{_to_float(item.get('attempt_pass_rate', 0.0)) * 100.0:.1f}%</td>"
            f"<td>{_to_float(item.get('mean_reward', 0.0)):.3f}</td>"
            f"<td>{_esc(_error_label(str(item.get('dominant_failure', ''))))}</td>"
            "</tr>"
        )

    failure_svg = ""
    failure_note = ""
    if failure_counts:
        sorted_failures = [item for item in failure_counts.most_common(8) if item[0] != "passed"]
        if sorted_failures:
            failure_svg = _bar_chart(
                [_error_label(kind) for kind, _ in sorted_failures],
                [_to_float(count) for _, count in sorted_failures],
                width=760,
                height=240,
                bar_color="#c4513f",
                label_color="#697577",
                grid_color="#eef1e8",
            )
    if not failure_svg:
        failure_note = '<div class="sub">No captured failures. Enable <code>capture_attempts</code> to inspect verification behavior.</div>'

    trace_sections_html: list[str] = []
    for idx, section in enumerate(trace_sections):
        tasks = list(section.get("tasks", []))
        shown_tasks = tasks[:12]
        hidden_count = max(0, len(tasks) - len(shown_tasks))

        task_blocks: list[str] = []
        for task in shown_tasks:
            attempt_rows = list(task.get("attempt_rows", []))
            attempt_lines = []
            for attempt in attempt_rows:
                status = "PASS" if bool(attempt.get("passed", False)) else "FAIL"
                badge_class = "pill-ok" if status == "PASS" else "pill-bad"
                attempt_lines.append(
                    "<tr>"
                    f"<td>{_to_int(attempt.get('idx', 0))}</td>"
                    f"<td><span class=\"pill {badge_class}\">{status}</span></td>"
                    f"<td>{_to_int(attempt.get('passed_cases', 0))}/{_to_int(attempt.get('total_cases', 0))}</td>"
                    f"<td>{_to_float(attempt.get('reward', 0.0)):.3f}</td>"
                    f"<td>{_to_float(attempt.get('runtime_ms', 0.0)):.1f}</td>"
                    f"<td>{_esc(attempt.get('detail', '-'))}</td>"
                    "</tr>"
                )

            summary = (
                f"{_esc(task.get('task_id', ''))} · "
                f"{_esc(task.get('difficulty', ''))} · "
                f"pass {_to_int(task.get('attempt_passed', 0))}/{_to_int(task.get('attempt_total', 0))} · "
                f"mean reward {_to_float(task.get('mean_reward', 0.0)):.3f}"
            )
            first_failure = _esc(task.get("first_failure", "-"))
            task_blocks.append(
                "<details class=\"trace-task\">"
                f"<summary>{summary}</summary>"
                f"<div class=\"trace-note\">First failure signal: <code>{first_failure}</code></div>"
                "<table class=\"trace-table\">"
                "<thead><tr><th>#</th><th>Status</th><th>Cases</th><th>Reward</th><th>Runtime ms</th><th>Detail</th></tr></thead>"
                f"<tbody>{''.join(attempt_lines)}</tbody>"
                "</table>"
                "</details>"
            )

        extra_note = ""
        if hidden_count > 0:
            extra_note = f'<div class="sub" style="margin-top:8px">Showing first {len(shown_tasks)} tasks for readability; {hidden_count} more tasks are in the JSON artifact.</div>'

        open_attr = " open" if idx == 0 else ""
        trace_sections_html.append(
            f"<details class=\"trace-run\"{open_attr}>"
            f"<summary>{_esc(section.get('model', ''))} / {_esc(section.get('strategy', ''))}"
            f"<span class=\"trace-meta\">attempt pass rate {_to_float(section.get('attempt_pass_rate', 0.0)) * 100.0:.1f}% ({_to_int(section.get('attempt_passed', 0))}/{_to_int(section.get('attempt_total', 0))})</span></summary>"
            f"{''.join(task_blocks)}"
            f"{extra_note}"
            "</details>"
        )

    verify_block = ""
    if verify_summaries:
        verify_block = (
            "<div class=\"grid2\">"
            "<div class=\"card\">"
            "<h2>Verification Summary</h2>"
            "<div class=\"sub\">Attempt-level verifier performance across model/strategy runs.</div>"
            "<table>"
            "<thead><tr><th>Model</th><th>Strategy</th><th>Attempts</th><th>Passed</th><th>Pass Rate</th><th>Mean Reward</th><th>Dominant Failure</th></tr></thead>"
            f"<tbody>{''.join(verify_rows)}</tbody>"
            "</table>"
            "</div>"
            "<div class=\"card\">"
            "<h2>Failure-Type Breakdown</h2>"
            f"{failure_note}"
            f"<svg viewBox=\"0 0 760 240\" preserveAspectRatio=\"none\">{failure_svg}</svg>"
            "</div>"
            "</div>"
            "<div class=\"card\">"
            "<h2>Verification Trace Explorer</h2>"
            "<div class=\"sub\">Open each run and task to inspect PASS/FAIL, cases, reward, runtime, and failure detail per attempt.</div>"
            f"{''.join(trace_sections_html)}"
            "</div>"
        )
    else:
        verify_block = (
            "<div class=\"card\">"
            "<h2>Verification Trace Explorer</h2>"
            "<div class=\"sub\">No attempt-level data in this payload. Re-run eval with <code>capture_attempts</code> enabled.</div>"
            "</div>"
        )

    html_doc = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{_esc(title)}</title>
  <style>
    :root {{
      --bg: #f6f4ee;
      --bg-soft: #eef6f2;
      --panel: #fffdf8;
      --line: #d7dacb;
      --text: #1f2a2e;
      --muted: #5e6c70;
      --teal: #1f8a8a;
      --orange: #e67e45;
      --green: #2f9f74;
      --red: #c4513f;
      --shadow: rgba(45, 55, 48, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 20px;
      color: var(--text);
      background:
        radial-gradient(circle at 12% 0%, #fde6cd 0%, transparent 32%),
        radial-gradient(circle at 88% 14%, #d6f1ea 0%, transparent 36%),
        var(--bg);
      font-family: "Space Grotesk", "Avenir Next", "Trebuchet MS", sans-serif;
    }}
    .wrap {{ max-width: 1360px; margin: 0 auto; display: grid; gap: 14px; }}
    .card {{
      background: linear-gradient(160deg, #fffefb, var(--panel));
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 7px 24px var(--shadow);
    }}
    h1, h2, h3 {{ margin: 0 0 8px 0; letter-spacing: 0.2px; }}
    h1 {{ font-size: 30px; }}
    h2 {{ font-size: 21px; }}
    h3 {{ font-size: 16px; }}
    .sub {{ color: var(--muted); font-size: 13px; }}
    .hero {{
      background: linear-gradient(120deg, #fff3df, #ecf8f4);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 7px 24px var(--shadow);
    }}
    .badge {{
      display: inline-block;
      margin-top: 8px;
      border-radius: 999px;
      padding: 5px 10px;
      background: #e6f4f2;
      color: #0d6b6b;
      font-size: 12px;
      border: 1px solid #b9dfd7;
    }}
    .kpis {{ display: grid; grid-template-columns: repeat(5, minmax(110px, 1fr)); gap: 10px; margin-top: 12px; }}
    .kpi {{ border: 1px solid var(--line); border-radius: 10px; padding: 10px; background: #fff; }}
    .kpi .v {{ font-size: 22px; font-weight: 700; color: #113f4a; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    svg {{ width: 100%; height: auto; background: #fff; border: 1px solid #e6eadf; border-radius: 10px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #e8ecdf; padding: 8px 6px; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ color: #3f5359; font-weight: 700; background: #f9fbf6; }}
    .trace-run {{ border: 1px solid #dde4d3; border-radius: 10px; background: #fff; padding: 8px 10px; margin-bottom: 10px; }}
    .trace-run > summary {{ cursor: pointer; font-weight: 700; color: #19464f; }}
    .trace-meta {{ margin-left: 8px; font-size: 12px; color: #47626a; font-weight: 500; }}
    .trace-task {{ border: 1px solid #e4eadc; border-radius: 10px; background: #fffcf6; padding: 8px; margin-top: 8px; }}
    .trace-task > summary {{ cursor: pointer; color: #2f4f55; font-size: 13px; }}
    .trace-note {{ margin: 8px 0 6px 0; font-size: 12px; color: #55666a; }}
    .trace-table th, .trace-table td {{ font-size: 12px; }}
    .pill {{ display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 11px; border: 1px solid transparent; }}
    .pill-ok {{ background: #def4e6; color: #176547; border-color: #b7dfc6; }}
    .pill-bad {{ background: #fde7e4; color: #8f2d21; border-color: #efb8b0; }}
    code {{ background: #fdf1dd; border: 1px solid #ecd8b3; border-radius: 6px; padding: 1px 6px; }}
    @media (max-width: 980px) {{
      .grid2 {{ grid-template-columns: 1fr; }}
      .kpis {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"hero\">
      <h1>TinyCodeTest Eval Report</h1>
      <div class=\"sub\">Generated (UTC): {_esc(payload.get('timestamp_utc', ''))}</div>
      <div class=\"badge\">Best run: {_esc(best.get('model', ''))} / {_esc(best.get('strategy', ''))}</div>
      <div class=\"kpis\">
        <div class=\"kpi\"><div class=\"sub\">Tasks</div><div class=\"v\">{_to_int(best.get('num_tasks', 0))}</div></div>
        <div class=\"kpi\"><div class=\"sub\">{_esc(pass_key)}</div><div class=\"v\">{_to_float(dict(best.get('overall', {})).get(pass_key, 0.0)):.3f}</div></div>
        <div class=\"kpi\"><div class=\"sub\">pass@{ks[-1]}</div><div class=\"v\">{_to_float(dict(best.get('overall', {})).get(f'pass@{ks[-1]}', 0.0)):.3f}</div></div>
        <div class=\"kpi\"><div class=\"sub\">Mean Reward</div><div class=\"v\">{_to_float(dict(best.get('overall', {})).get('mean_reward', 0.0)):.3f}</div></div>
        <div class=\"kpi\"><div class=\"sub\">Avg Runtime (ms)</div><div class=\"v\">{_to_float(dict(best.get('overall', {})).get('avg_runtime_ms', 0.0)):.1f}</div></div>
      </div>
    </div>

    <div class=\"grid2\">
      <div class=\"card\">
        <h2>Reward Curve (Best Run)</h2>
        <div class=\"sub\">Mean reward per task across sampled completions.</div>
        <svg viewBox=\"0 0 760 260\" preserveAspectRatio=\"none\">
          <line x1=\"16\" y1=\"244\" x2=\"744\" y2=\"244\" stroke=\"#e6eadf\" stroke-width=\"1\" />
          <line x1=\"16\" y1=\"138\" x2=\"744\" y2=\"138\" stroke=\"#eef1e8\" stroke-width=\"1\" />
          <line x1=\"16\" y1=\"32\" x2=\"744\" y2=\"32\" stroke=\"#eef1e8\" stroke-width=\"1\" />
          <polyline fill=\"none\" stroke=\"#1f8a8a\" stroke-width=\"2.5\" points=\"{reward_points}\" />
        </svg>
      </div>
      <div class=\"card\">
        <h2>Reward Distribution</h2>
        <div class=\"sub\">Histogram of best-run mean task rewards.</div>
        <svg viewBox=\"0 0 760 240\" preserveAspectRatio=\"none\">{hist_svg}</svg>
      </div>
    </div>

    <div class=\"grid2\">
      <div class=\"card\">
        <h2>Run Comparison ({_esc(pass_key)})</h2>
        <div class=\"sub\">Higher bars indicate stronger overall pass rate.</div>
        <svg viewBox=\"0 0 760 240\" preserveAspectRatio=\"none\">{model_svg}</svg>
      </div>
      <div class=\"card\">
        <h2>Difficulty Split ({_esc(pass_key)}, Best Run)</h2>
        <div class=\"sub\">Pass rate split for easy, medium, and hard subsets.</div>
        <svg viewBox=\"0 0 420 240\" preserveAspectRatio=\"none\">{diff_svg}</svg>
      </div>
    </div>

    <div class=\"card\">
      <h2>Leaderboard Snapshot</h2>
      <table>
        <thead><tr><th>#</th><th>Model</th><th>Strategy</th><th>{_esc(pass_key)}</th><th>pass@{ks[-1]}</th><th>Adv {_esc(pass_key)}</th><th>Mean Reward</th><th>Avg Runtime ms</th><th>Attempt Pass Rate</th></tr></thead>
        <tbody>
          {''.join(run_rows)}
        </tbody>
      </table>
    </div>

    {verify_block}
  </div>
</body>
</html>
"""
    return html_doc


def save_html_report(
    payload: dict[str, object],
    output_path: str | Path,
    *,
    title: str = "TinyCodeTest Report",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report_html(payload, title=title), encoding="utf-8")
    return path


def load_eval_payload(path: str | Path) -> dict[str, object]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))
