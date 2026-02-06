from __future__ import annotations

import json
from pathlib import Path


def _sort_runs(payload: dict[str, object]) -> list[dict[str, object]]:
    config = dict(payload.get("config", {}))
    ks = sorted(int(value) for value in config.get("ks", [1]))
    primary = f"pass@{ks[0]}"
    secondary = f"pass@{ks[-1]}"
    runs = list(payload.get("runs", []))
    runs.sort(
        key=lambda run: (
            float(dict(run.get("overall", {})).get(primary, 0.0)),
            float(dict(run.get("overall", {})).get(secondary, 0.0)),
        ),
        reverse=True,
    )
    return runs


def _polyline(values: list[float], width: int, height: int, pad: int = 8) -> str:
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


def _bar_chart(labels: list[str], values: list[float], width: int, height: int, pad: int = 8) -> str:
    if not values:
        return ""
    top = max(values)
    if top <= 0:
        top = 1e-9

    n = len(values)
    gap = 4
    usable = width - (2 * pad)
    bar_w = max(2, int((usable - gap * (n - 1)) / max(1, n)))

    parts: list[str] = []
    for idx, value in enumerate(values):
        x = pad + idx * (bar_w + gap)
        bar_h = ((value / top) * (height - (2 * pad))) if value > 0 else 0.0
        y = height - pad - bar_h
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="2" fill="#7d6aff" />'
        )
        label = labels[idx][:12]
        parts.append(
            f'<text x="{(x + bar_w / 2):.1f}" y="{height - 1}" fill="#a0a0a0" font-size="9" text-anchor="middle">{label}</text>'
        )
    return "".join(parts)


def render_report_html(payload: dict[str, object], *, title: str = "TinyCodeTest Report") -> str:
    runs = _sort_runs(payload)
    if not runs:
        raise ValueError("No runs available in payload.")

    best = runs[0]
    best_rows = list(best.get("rows", []))
    reward_series = [float(dict(row).get("mean_reward", 0.0)) for row in best_rows]
    reward_points = _polyline(reward_series, width=680, height=240)

    bins, counts = _histogram(reward_series, bins=10)
    hist_svg = _bar_chart(bins, [float(x) for x in counts], width=680, height=220)

    config = dict(payload.get("config", {}))
    ks = sorted(int(value) for value in config.get("ks", [1]))
    pass_key = f"pass@{ks[0]}"
    model_labels = [f"{run['model']}\n{run['strategy']}" for run in runs]
    model_scores = [float(dict(run.get("overall", {})).get(pass_key, 0.0)) for run in runs]
    model_svg = _bar_chart(model_labels, model_scores, width=680, height=220)

    difficulty = dict(best.get("difficulty", {}))
    diff_labels = ["easy", "medium", "hard"]
    diff_scores = [float(dict(difficulty.get(lbl, {})).get(pass_key, 0.0)) for lbl in diff_labels]
    diff_svg = _bar_chart(diff_labels, diff_scores, width=360, height=220)

    run_rows = []
    for idx, run in enumerate(runs, start=1):
        overall = dict(run.get("overall", {}))
        run_rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{run.get('model', '')}</td>"
            f"<td>{run.get('strategy', '')}</td>"
            f"<td>{float(overall.get(pass_key, 0.0)):.3f}</td>"
            f"<td>{float(overall.get(f'pass@{ks[-1]}', 0.0)):.3f}</td>"
            f"<td>{float(overall.get('mean_reward', 0.0)):.3f}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: #060606;
      --panel: #101010;
      --text: #f4f4f4;
      --muted: #a3a3a3;
      --accent: #7d6aff;
      --line: #232323;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 20px; background: radial-gradient(circle at top left, #151515, var(--bg)); color: var(--text); font-family: "IBM Plex Sans", "SF Pro Display", sans-serif; }}
    .wrap {{ max-width: 1320px; margin: 0 auto; display: grid; gap: 14px; }}
    .card {{ background: linear-gradient(165deg, #121212, #090909); border: 1px solid var(--line); border-radius: 14px; padding: 14px; }}
    h1, h2 {{ margin: 0 0 10px 0; letter-spacing: 0.2px; }}
    .sub {{ color: var(--muted); font-size: 13px; margin-bottom: 6px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; }}
    .kpi {{ border: 1px solid var(--line); border-radius: 10px; padding: 10px; background: #0c0c0c; }}
    .kpi .v {{ font-size: 22px; font-weight: 700; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    svg {{ width: 100%; height: auto; background: #0b0b0b; border: 1px solid var(--line); border-radius: 10px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; font-size: 13px; }}
    th {{ color: #cccccc; font-weight: 600; }}
    .badge {{ display: inline-block; background: #191432; color: #c8bcff; padding: 4px 8px; border-radius: 999px; font-size: 12px; }}
    @media (max-width: 960px) {{
      .grid2 {{ grid-template-columns: 1fr; }}
      .kpis {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"card\">
      <h1>TinyCodeTest Eval Report</h1>
      <div class=\"sub\">Generated (UTC): {payload.get('timestamp_utc', '')}</div>
      <div class=\"badge\">Best Run: {best.get('model', '')} / {best.get('strategy', '')}</div>
      <div class=\"kpis\" style=\"margin-top:10px\">
        <div class=\"kpi\"><div class=\"sub\">Tasks</div><div class=\"v\">{best.get('num_tasks', 0)}</div></div>
        <div class=\"kpi\"><div class=\"sub\">{pass_key}</div><div class=\"v\">{float(dict(best.get('overall', {})).get(pass_key, 0.0)):.3f}</div></div>
        <div class=\"kpi\"><div class=\"sub\">pass@{ks[-1]}</div><div class=\"v\">{float(dict(best.get('overall', {})).get(f'pass@{ks[-1]}', 0.0)):.3f}</div></div>
        <div class=\"kpi\"><div class=\"sub\">Mean Reward</div><div class=\"v\">{float(dict(best.get('overall', {})).get('mean_reward', 0.0)):.3f}</div></div>
      </div>
    </div>

    <div class=\"grid2\">
      <div class=\"card\">
        <h2>Reward Curve (Best Run)</h2>
        <svg viewBox=\"0 0 680 240\" preserveAspectRatio=\"none\">
          <polyline fill=\"none\" stroke=\"#7d6aff\" stroke-width=\"2\" points=\"{reward_points}\" />
        </svg>
      </div>
      <div class=\"card\">
        <h2>Latest Reward Distribution</h2>
        <svg viewBox=\"0 0 680 220\" preserveAspectRatio=\"none\">{hist_svg}</svg>
      </div>
    </div>

    <div class=\"grid2\">
      <div class=\"card\">
        <h2>Run Comparison ({pass_key})</h2>
        <svg viewBox=\"0 0 680 220\" preserveAspectRatio=\"none\">{model_svg}</svg>
      </div>
      <div class=\"card\">
        <h2>Difficulty Split ({pass_key}, Best Run)</h2>
        <svg viewBox=\"0 0 360 220\" preserveAspectRatio=\"none\">{diff_svg}</svg>
      </div>
    </div>

    <div class=\"card\">
      <h2>Leaderboard Snapshot</h2>
      <table>
        <thead><tr><th>#</th><th>Model</th><th>Strategy</th><th>{pass_key}</th><th>pass@{ks[-1]}</th><th>Mean Reward</th></tr></thead>
        <tbody>
          {''.join(run_rows)}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""
    return html


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
