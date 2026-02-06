#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cgi
import html
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinycodetest.dataset import TaskSampler, dataset_summary, generate_and_write, load_dataset
from tinycodetest.eval_runner import EvalConfig, evaluate_suite, save_eval
from tinycodetest.harness import PromptStrategy, SandboxConfig
from tinycodetest.models import builtin_models, resolve_models
from tinycodetest.reporting import save_html_report

IS_VERCEL = bool(os.getenv("VERCEL"))
STORAGE_ROOT = Path("/tmp/tinycodetest_web") if IS_VERCEL else ROOT
UPLOAD_DIR = STORAGE_ROOT / "data" / "uploads"
MODEL_CONFIG_UPLOAD_DIR = STORAGE_ROOT / "configs" / "uploads"
WEB_RUN_DIR = STORAGE_ROOT / "results" / "web_runs"
RUN_META_DIR = STORAGE_ROOT / "results" / "run_meta"

RUNS: dict[str, dict[str, Any]] = {}


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_path_from_input(value: str, *, default: Path | None = None) -> Path:
    text = value.strip()
    if not text:
        if default is None:
            raise ValueError("Path value cannot be empty")
        return default
    path = Path(text)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _coerce_int(value: str, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _coerce_float(value: str, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _save_uploaded_file(field: cgi.FieldStorage, target_dir: Path, suffix: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{_now_stamp()}_{uuid4().hex[:8]}{suffix}"
    out_path = target_dir / safe_name

    with out_path.open("wb") as handle:
        data = field.file.read()
        handle.write(data)
    return out_path


def _run_meta_path(run_id: str) -> Path:
    return RUN_META_DIR / f"{run_id}.json"


def _load_run(run_id: str) -> dict[str, Any] | None:
    if run_id in RUNS:
        return RUNS[run_id]
    path = _run_meta_path(run_id)
    if not path.exists():
        return None
    try:
        payload = dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None
    RUNS[run_id] = payload
    return payload


def _save_run(run_id: str, payload: dict[str, Any]) -> None:
    RUNS[run_id] = payload
    RUN_META_DIR.mkdir(parents=True, exist_ok=True)
    _run_meta_path(run_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _list_runs() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if RUN_META_DIR.exists():
        for path in RUN_META_DIR.glob("*.json"):
            try:
                payload = dict(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
            run_id = path.stem
            records[run_id] = payload
    records.update(RUNS)
    return records


def _render_layout(title: str, body: str) -> bytes:
    doc = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f7f3ea;
      --bg-soft: #efe7d4;
      --panel: #fffdf8;
      --line: #ddcfb0;
      --text: #251e13;
      --muted: #68553b;
      --accent: #d95d39;
      --accent2: #2f9e8f;
      --warn: #b35a00;
      --ok: #1f7a67;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Space Grotesk", "Avenir Next", "Trebuchet MS", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 10% 0%, #ffdcb8 0%, transparent 35%),
        radial-gradient(circle at 95% 20%, #d2f3ef 0%, transparent 40%),
        var(--bg);
      min-height: 100vh;
    }}
    .wrap {{ max-width: 1300px; margin: 0 auto; padding: 18px; display: grid; gap: 14px; }}
    .hero {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: linear-gradient(120deg, #fff4e8, #eefaf8);
      padding: 16px;
    }}
    h1 {{ margin: 0 0 8px 0; font-size: 30px; letter-spacing: 0.2px; }}
    h2 {{ margin: 0 0 8px 0; font-size: 20px; }}
    p {{ margin: 0; color: var(--muted); }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: linear-gradient(165deg, #fffefb, var(--panel));
      padding: 14px;
      box-shadow: 0 6px 20px rgba(55, 38, 17, 0.06);
    }}
    .grid2 {{ display: grid; grid-template-columns: 1.5fr 1fr; gap: 12px; }}
    .grid22 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .grid3 {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px; }}
    .row3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 10px; }}
    label {{ display: block; font-size: 13px; color: #4f3f2d; margin-bottom: 6px; }}
    input[type=text], input[type=number], textarea, select {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid #d9c7a2;
      padding: 9px 10px;
      background: #fffefb;
      color: var(--text);
      font-size: 14px;
    }}
    textarea {{ min-height: 68px; resize: vertical; }}
    .checks {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; border: 1px solid #d9c7a2; border-radius: 10px; padding: 8px; background: #fffaf1; }}
    .check {{ font-size: 13px; color: #3b2f21; display: flex; gap: 8px; align-items: center; }}
    .actions {{ display: flex; gap: 10px; align-items: center; margin-top: 10px; }}
    .btn {{
      border: none;
      border-radius: 10px;
      background: linear-gradient(120deg, var(--accent), #f0804f);
      color: white;
      padding: 10px 14px;
      font-size: 14px;
      cursor: pointer;
    }}
    .btn:hover {{ filter: brightness(1.08); }}
    .btn2 {{
      border: 1px solid #c8b187;
      border-radius: 10px;
      background: #fff7e9;
      color: #4a3722;
      padding: 10px 14px;
      font-size: 14px;
      text-decoration: none;
      display: inline-block;
    }}
    .small {{ font-size: 12px; color: var(--muted); }}
    .runlist {{ display: grid; gap: 8px; }}
    .run {{ border: 1px solid #d4c19b; border-radius: 10px; padding: 8px; background: #fffbf3; }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 8px;
      background: #fbe0d6;
      color: #8a391f;
      font-size: 12px;
      margin-right: 6px;
    }}
    .ok {{ background: #daf2e8; color: #17604f; }}
    .warn {{ background: #fde8c9; color: #8a4e00; }}
    .links a {{ color: #0e796c; text-decoration: none; margin-right: 10px; }}
    .links a:hover {{ text-decoration: underline; }}
    .error {{ border: 1px solid #d8897f; background: #fff2ef; color: #72261d; padding: 10px; border-radius: 10px; white-space: pre-wrap; }}
    iframe {{ width: 100%; min-height: 650px; border: 1px solid #d4c19b; border-radius: 10px; background: #fffefb; }}
    ul {{ margin: 6px 0; padding-left: 18px; }}
    code {{ background: #fff1de; padding: 1px 6px; border-radius: 6px; border: 1px solid #ecd6af; }}
    @media (max-width: 980px) {{
      .grid2 {{ grid-template-columns: 1fr; }}
      .grid22 {{ grid-template-columns: 1fr; }}
      .row, .row3 {{ grid-template-columns: 1fr; }}
      .grid3 {{ grid-template-columns: 1fr; }}
      .checks {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"hero\">
      <h1>TinyCodeTest Web Runner</h1>
      <p>Local tool: upload dataset, pick models, run eval, and open the generated report.</p>
    </div>
    {body}
  </div>
</body>
</html>
"""
    return doc.encode("utf-8")


def _list_recent_runs() -> str:
    all_runs = _list_runs()
    if not all_runs:
        return '<div class="small">No runs yet.</div>'

    items = []
    for run_id, run in sorted(all_runs.items(), key=lambda pair: pair[1].get("started", ""), reverse=True)[:12]:
        status = str(run.get("status", ""))
        badge_cls = "ok" if status == "completed" else "warn"
        summary = html.escape(str(run.get("summary", "")))
        items.append(
            "<div class=\"run\">"
            f"<div><span class=\"badge {badge_cls}\">{html.escape(status)}</span><strong>{html.escape(run_id)}</strong></div>"
            f"<div class=\"small\">{summary}</div>"
            f"<div class=\"links\" style=\"margin-top:6px\"><a href=\"/run/{html.escape(run_id)}\">Open</a></div>"
            "</div>"
        )
    return "".join(items)


def _index_body(error: str = "") -> str:
    builtin = sorted(builtin_models().keys())
    default_dataset = (
        str((ROOT / "data" / "tinycodetest.jsonl").resolve())
        if IS_VERCEL
        else "data/tinycodetest.jsonl"
    )
    default_samples = "2" if IS_VERCEL else "5"
    default_max_tasks = "8" if IS_VERCEL else "40"
    default_timeout = "0.6" if IS_VERCEL else "0.8"
    deployment_note = (
        "This deployment is running on Vercel serverless. Files are temporary and run history may reset after cold starts."
        if IS_VERCEL
        else "This web app runs locally. It does not change your Vercel deployment URL."
    )
    model_checks = "".join(
        (
            f'<label class="check"><input type="checkbox" name="model" value="{html.escape(name)}" ' +
            ("checked" if name in {"heuristic-small", "heuristic-medium"} else "") +
            f' /> {html.escape(name)}</label>'
        )
        for name in builtin
    )
    strategy_checks = "".join(
        (
            f'<label class="check"><input type="checkbox" name="strategy" value="{html.escape(name)}" ' +
            ("checked" if name == "direct" else "") +
            f' /> {html.escape(name)}</label>'
        )
        for name in ["direct", "plan_then_code"]
    )

    err_block = f'<div class="error">{html.escape(error)}</div>' if error else ""
    return f"""
<div class=\"grid2\">
  <div class=\"card\">
    <h2>Run Evaluation</h2>
    {err_block}
    <form method=\"post\" action=\"/run\" enctype=\"multipart/form-data\">
      <div class=\"row\">
        <div>
          <label>Dataset Path (existing JSONL)</label>
          <input type=\"text\" name=\"dataset_path\" value=\"{html.escape(default_dataset)}\" />
        </div>
        <div>
          <label>Or Upload Dataset (.jsonl)</label>
          <input type=\"file\" name=\"dataset_file\" accept=\".jsonl\" />
        </div>
      </div>

      <div class=\"row\">
        <div>
          <label>Model Config Path (optional)</label>
          <input type=\"text\" name=\"model_config_path\" placeholder=\"configs/models.example.json\" />
        </div>
        <div>
          <label>Or Upload Model Config (.json)</label>
          <input type=\"file\" name=\"model_config_file\" accept=\".json\" />
        </div>
      </div>

      <div class=\"row\">
        <div>
          <label>Built-in Models</label>
          <div class=\"checks\">{model_checks}</div>
        </div>
        <div>
          <label>Extra Model Names (comma-separated; from model config)</label>
          <input type=\"text\" name=\"extra_models\" placeholder=\"qwen-small-local,deepseek-medium-hosted\" />
        </div>
      </div>

      <div class=\"row\">
        <div>
          <label>Prompt Strategies</label>
          <div class=\"checks\">{strategy_checks}</div>
        </div>
        <div>
          <label>pass@k Values</label>
          <input type=\"text\" name=\"ks\" value=\"1,5\" />
        </div>
      </div>

      <div class=\"row3\">
        <div><label>Samples / Task</label><input type=\"number\" name=\"samples_per_task\" value=\"{default_samples}\" min=\"1\" /></div>
        <div><label>Max Tasks (0 = all)</label><input type=\"number\" name=\"max_tasks\" value=\"{default_max_tasks}\" min=\"0\" /></div>
        <div><label>Seed</label><input type=\"number\" name=\"seed\" value=\"13\" /></div>
      </div>

      <div class=\"row3\">
        <div><label>Timeout (seconds)</label><input type=\"number\" step=\"0.1\" name=\"timeout\" value=\"{default_timeout}\" min=\"0.1\" /></div>
        <div><label>Memory MB</label><input type=\"number\" name=\"memory_mb\" value=\"256\" min=\"64\" /></div>
        <div><label>Stem (optional)</label><input type=\"text\" name=\"stem\" placeholder=\"web_run_custom\" /></div>
      </div>

      <div class=\"actions\">
        <button type=\"submit\" class=\"btn\">Run Evaluation</button>
        <span class=\"small\">JSON + leaderboard + HTML report are generated automatically.</span>
      </div>
    </form>
  </div>

  <div class=\"card\">
    <h2>Where To Find Things</h2>
    <div class=\"small\">{html.escape(deployment_note)}</div>
    <ul>
      <li>Dataset uploads: <code>data/uploads/</code></li>
      <li>Model-config uploads: <code>configs/uploads/</code></li>
      <li>Run artifacts: <code>results/web_runs/</code></li>
    </ul>
  </div>
</div>

<div class=\"grid22\">
  <div class=\"card\">
    <h2>What It Does</h2>
    <ul>
      <li>Upload dataset JSONL or use existing dataset path</li>
      <li>Upload model-config JSON or use existing model-config path</li>
      <li>Pick models and prompt strategies</li>
      <li>Run evaluation from browser</li>
      <li>Open JSON, Markdown, and HTML report outputs</li>
      <li>Preview report directly in the run page</li>
    </ul>
  </div>
  <div class=\"card\">
    <h2>Recent Runs</h2>
    <div class=\"runlist\">{_list_recent_runs()}</div>
    <div class=\"small\" style=\"margin-top:8px\">Artifacts are saved under <code>results/web_runs/</code>.</div>
  </div>
</div>
"""


def _run_page(run_id: str) -> tuple[int, bytes]:
    run = _load_run(run_id)
    if run is None:
        body = '<div class="card"><h2>Run Not Found</h2><p>Unknown run id.</p><a class="btn2" href="/">Back</a></div>'
        return 404, _render_layout("Run Not Found", body)

    status = html.escape(str(run.get("status", "unknown")))
    summary = html.escape(str(run.get("summary", "")))
    err = run.get("error", "")
    err_block = f'<div class="error" style="margin-top:10px">{html.escape(str(err))}</div>' if err else ""

    links = []
    for kind in ["json", "md", "html"]:
        if kind in run:
            links.append(f'<a href="/artifact/{html.escape(run_id)}/{kind}" target="_blank">{kind.upper()}</a>')

    iframe = ""
    if "html" in run:
        iframe = f'<div class="card"><h2>Report Preview</h2><iframe src="/artifact/{html.escape(run_id)}/html"></iframe></div>'

    path_lines = []
    for kind in ["json", "md", "html"]:
        if kind in run:
            path_lines.append(f"<li><code>{html.escape(str(run[kind]))}</code></li>")
    paths_block = f"<ul>{''.join(path_lines)}</ul>" if path_lines else '<div class="small">No artifacts yet.</div>'

    body = f"""
<div class=\"card\">
  <h2>Run: {html.escape(run_id)}</h2>
  <div><span class=\"badge {'ok' if status == 'completed' else 'warn'}\">{status}</span></div>
  <div class=\"small\" style=\"margin-top:6px\">{summary}</div>
  <div class=\"small\" style=\"margin-top:6px\">Artifacts directory: <code>{html.escape(str(WEB_RUN_DIR))}</code></div>
  <div class=\"links\" style=\"margin-top:10px\">{' '.join(links)}</div>
  <div class=\"small\" style=\"margin-top:6px\">Artifact Paths:</div>
  {paths_block}
  {err_block}
  <div class=\"actions\"><a class=\"btn2\" href=\"/\">Back</a></div>
</div>
{iframe}
"""
    return 200, _render_layout(f"Run {run_id}", body)


class AppHandler(BaseHTTPRequestHandler):
    server_version = "TinyCodeTestWeb/0.1"

    def _send_bytes(self, status: int, payload: bytes, content_type: str = "text/html; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _redirect(self, location: str) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._send_bytes(200, _render_layout("TinyCodeTest Web", _index_body()))
            return

        if path.startswith("/run/"):
            run_id = path.split("/", 2)[2]
            status, payload = _run_page(run_id)
            self._send_bytes(status, payload)
            return

        if path.startswith("/artifact/"):
            parts = path.split("/")
            if len(parts) != 4:
                self._send_bytes(404, b"Not found", "text/plain; charset=utf-8")
                return
            _, _, run_id, kind = parts
            run = _load_run(run_id)
            if run is None or kind not in run:
                self._send_bytes(404, b"Not found", "text/plain; charset=utf-8")
                return
            file_path = Path(str(run[kind]))
            if not file_path.exists():
                self._send_bytes(404, b"Missing artifact", "text/plain; charset=utf-8")
                return
            content = file_path.read_bytes()
            if kind == "json":
                ctype = "application/json; charset=utf-8"
            elif kind == "md":
                ctype = "text/markdown; charset=utf-8"
            else:
                ctype = "text/html; charset=utf-8"
            self._send_bytes(200, content, ctype)
            return

        if path == "/favicon.ico":
            self._send_bytes(204, b"", "image/x-icon")
            return

        self._send_bytes(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/run":
            self._send_bytes(404, b"Not found", "text/plain; charset=utf-8")
            return

        try:
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" in content_type:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": content_type,
                    },
                )
                run_id = self._handle_run_from_form(form)
                self._redirect(f"/run/{run_id}")
                return

            content_length = int(self.headers.get("Content-Length", "0") or "0")
            data = self.rfile.read(content_length).decode("utf-8", errors="replace")
            params = {k: v[0] for k, v in parse_qs(data).items()}
            run_id = self._handle_run_from_query(params)
            self._redirect(f"/run/{run_id}")
        except Exception as exc:
            trace = traceback.format_exc(limit=8)
            message = f"{exc}\n\n{trace}"
            self._send_bytes(400, _render_layout("Run Failed", _index_body(error=message)))

    def _handle_run_from_query(self, params: dict[str, str]) -> str:
        # Fallback for non-multipart posts; mostly for debugging.
        fake = {
            "dataset_path": params.get("dataset_path", "data/tinycodetest.jsonl"),
            "model_config_path": params.get("model_config_path", ""),
            "extra_models": params.get("extra_models", ""),
            "ks": params.get("ks", "1,5"),
            "samples_per_task": params.get("samples_per_task", "5"),
            "max_tasks": params.get("max_tasks", "40"),
            "seed": params.get("seed", "13"),
            "timeout": params.get("timeout", "0.8"),
            "memory_mb": params.get("memory_mb", "256"),
            "stem": params.get("stem", ""),
            "models": params.get("models", "heuristic-small"),
            "strategies": params.get("strategies", "direct"),
        }
        model_names = _parse_csv(fake["models"])
        strategy_names = _parse_csv(fake["strategies"])
        return self._run_eval_job(
            dataset_path=_safe_path_from_input(fake["dataset_path"]),
            model_config_path=_safe_path_from_input(fake["model_config_path"]) if fake["model_config_path"].strip() else None,
            model_names=model_names,
            extra_models=_parse_csv(fake["extra_models"]),
            strategy_names=strategy_names,
            ks_text=fake["ks"],
            samples_per_task_text=fake["samples_per_task"],
            max_tasks_text=fake["max_tasks"],
            seed_text=fake["seed"],
            timeout_text=fake["timeout"],
            memory_mb_text=fake["memory_mb"],
            stem_text=fake["stem"],
        )

    def _handle_run_from_form(self, form: cgi.FieldStorage) -> str:
        dataset_upload = form["dataset_file"] if "dataset_file" in form else None
        dataset_path_text = form.getfirst("dataset_path", "data/tinycodetest.jsonl")

        if dataset_upload is not None and getattr(dataset_upload, "filename", ""):
            dataset_path = _save_uploaded_file(dataset_upload, UPLOAD_DIR, ".jsonl")
        else:
            dataset_path = _safe_path_from_input(dataset_path_text)

        model_config_upload = form["model_config_file"] if "model_config_file" in form else None
        model_config_path_text = form.getfirst("model_config_path", "")
        if model_config_upload is not None and getattr(model_config_upload, "filename", ""):
            model_config_path = _save_uploaded_file(model_config_upload, MODEL_CONFIG_UPLOAD_DIR, ".json")
        elif model_config_path_text.strip():
            model_config_path = _safe_path_from_input(model_config_path_text)
        else:
            model_config_path = None

        model_names = [item for item in form.getlist("model") if item]
        strategy_names = [item for item in form.getlist("strategy") if item]

        return self._run_eval_job(
            dataset_path=dataset_path,
            model_config_path=model_config_path,
            model_names=model_names,
            extra_models=_parse_csv(form.getfirst("extra_models", "")),
            strategy_names=strategy_names,
            ks_text=form.getfirst("ks", "1,5"),
            samples_per_task_text=form.getfirst("samples_per_task", "5"),
            max_tasks_text=form.getfirst("max_tasks", "40"),
            seed_text=form.getfirst("seed", "13"),
            timeout_text=form.getfirst("timeout", "0.8"),
            memory_mb_text=form.getfirst("memory_mb", "256"),
            stem_text=form.getfirst("stem", ""),
        )

    def _run_eval_job(
        self,
        *,
        dataset_path: Path,
        model_config_path: Path | None,
        model_names: list[str],
        extra_models: list[str],
        strategy_names: list[str],
        ks_text: str,
        samples_per_task_text: str,
        max_tasks_text: str,
        seed_text: str,
        timeout_text: str,
        memory_mb_text: str,
        stem_text: str,
    ) -> str:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        if not model_names and not extra_models:
            model_names = ["heuristic-small"]
        model_names = model_names + [name for name in extra_models if name not in model_names]

        if not strategy_names:
            strategy_names = ["direct"]

        ks = tuple(int(value) for value in _parse_csv(ks_text or "1,5"))
        if not ks:
            ks = (1, 5)

        samples_per_task = max(1, _coerce_int(samples_per_task_text, 5))
        max_tasks = max(0, _coerce_int(max_tasks_text, 40))
        seed = _coerce_int(seed_text, 13)
        timeout = max(0.1, _coerce_float(timeout_text, 0.8))
        memory_mb = max(64, _coerce_int(memory_mb_text, 256))

        run_id = stem_text.strip() or f"web_{_now_stamp()}_{uuid4().hex[:6]}"

        _save_run(
            run_id,
            {
                "status": "running",
                "started": datetime.now(timezone.utc).isoformat(),
                "summary": f"dataset={dataset_path.name}; models={','.join(model_names)}",
            },
        )

        try:
            tasks = load_dataset(dataset_path)
            if max_tasks > 0 and max_tasks < len(tasks):
                tasks = TaskSampler(tasks).sample(n=max_tasks, seed=seed)

            strategies = [PromptStrategy(name) for name in strategy_names]
            models = resolve_models(
                model_names,
                model_config_path=str(model_config_path) if model_config_path else None,
            )

            payload = evaluate_suite(
                models=models,
                strategies=strategies,
                tasks=tasks,
                sandbox=SandboxConfig(timeout_seconds=timeout, memory_mb=memory_mb),
                config=EvalConfig(samples_per_task=samples_per_task, ks=ks, seed=seed, capture_attempts=False),
            )
            payload["config"]["model_config"] = str(model_config_path) if model_config_path else None
            payload["dataset"] = {
                "path": str(dataset_path),
                "selected_tasks": len(tasks),
                "summary": dataset_summary(tasks).__dict__,
            }

            json_path, md_path = save_eval(payload, WEB_RUN_DIR, run_id)
            html_path = save_html_report(payload, WEB_RUN_DIR / f"{run_id}.html")

            run_payload = _load_run(run_id) or {}
            run_payload.update(
                {
                    "status": "completed",
                    "json": str(json_path),
                    "md": str(md_path),
                    "html": str(html_path),
                    "summary": (
                        f"tasks={len(tasks)}; models={','.join(model_names)}; "
                        f"strategies={','.join(strategy_names)}; ks={','.join(str(k) for k in ks)}"
                    ),
                }
            )
            _save_run(run_id, run_payload)
        except Exception as exc:
            run_payload = _load_run(run_id) or {}
            run_payload.update(
                {
                    "status": "failed",
                    "error": f"{exc}\n\n{traceback.format_exc(limit=10)}",
                }
            )
            _save_run(run_id, run_payload)
            raise

        return run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TinyCodeTest local web app wrapper.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8787, help="Port to bind")
    parser.add_argument(
        "--auto-generate-dataset",
        action="store_true",
        help="Generate default dataset if data/tinycodetest.jsonl is missing",
    )
    parser.add_argument("--tasks", type=int, default=600, help="Task count for auto generation")
    parser.add_argument("--dataset-seed", type=int, default=7, help="Seed for auto generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_default = ROOT / "data" / "tinycodetest.jsonl"
    if args.auto_generate_dataset and not dataset_default.exists():
        generate_and_write(
            output_path=dataset_default,
            adversarial_output_path=ROOT / "data" / "tinycodetest_adversarial.jsonl",
            total_tasks=args.tasks,
            seed=args.dataset_seed,
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CONFIG_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    WEB_RUN_DIR.mkdir(parents=True, exist_ok=True)
    RUN_META_DIR.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"TinyCodeTest web app running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
