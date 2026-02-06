# TinyCodeTest: A Deterministic Verifiers-Style RL Environment

TinyCodeTest is a `verifiers`-style environment for Python code generation with deterministic rewards.
It ships an end-to-end benchmark stack:
- dataset sampler (difficulty buckets + adversarial split)
- prompt harness
- sandboxed deterministic verifier (unit tests + timeout)
- eval runner that writes leaderboard tables and JSON artifacts

## Why This Matters
This environment is designed as research infrastructure, not just a toy task:
- fully deterministic reward for RL and eval stability
- explicit difficulty buckets (`easy`, `medium`, `hard`)
- adversarial edge-case split to expose shallow heuristics
- benchmark artifacts (`pass@k`, rewards, JSON outputs) that can be compared run-to-run

## What Is Included
- `tinycodetest/dataset.py`: task templates, dataset generation, sampler
- `tinycodetest/harness.py`: prompt strategies (`direct`, `plan_then_code`)
- `tinycodetest/verifier.py`: sandboxed execution + deterministic scoring
- `tinycodetest/environment.py`: `TinyCodeTestEnvironment` (`sample`, `prompt`, `verify`)
- `tinycodetest/eval_runner.py`: `pass@k`, difficulty/adversarial metrics, leaderboard export
- `scripts/generate_dataset.py`: build dataset JSONL files
- `scripts/run_eval.py`: evaluate one or more model adapters and save JSON/MD
- `scripts/run_baselines.py`: convenience baseline run
- `scripts/generate_report.py`: generate an HTML dashboard report from eval JSON
- `scripts/validate_dataset.py`: validate dataset shape and optional canonical correctness
- `scripts/export_train_data.py`: export prompt/completion JSONL for fine-tuning
- `scripts/web_app.py`: local web UI (upload dataset, pick models, run eval, view report)
- `configs/models.example.json`: example multi-model API registry config
- `docs/one_pager.md`: benchmark one-pager (what it measures + failure modes)

## Quickstart

### 1) Generate Dataset (600 default tasks)
```bash
python3 scripts/generate_dataset.py \
  --output data/tinycodetest.jsonl \
  --adversarial-output data/tinycodetest_adversarial.jsonl \
  --tasks 600 \
  --seed 7
```

### 2) Run Eval (leaderboard + JSON)
```bash
python3 scripts/run_eval.py \
  --dataset data/tinycodetest.jsonl \
  --models heuristic-small,heuristic-medium,reference-oracle \
  --strategies direct,plan_then_code \
  --samples-per-task 5 \
  --ks 1,5 \
  --max-tasks 120 \
  --timeout 0.7 \
  --output-dir results \
  --generate-report \
  --stem baseline_120
```

### 3) Baseline Shortcut
```bash
python3 scripts/run_baselines.py --max-tasks 150 --samples-per-task 5 --include-oracle
```

### 4) Optional Real-Model Baseline (OpenAI-Compatible API)
```bash
export OPENAI_API_KEY=...
export TCT_OPENAI_BASE_URL=https://your-endpoint/v1
export TCT_OPENAI_MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct

python3 scripts/run_eval.py \
  --dataset data/tinycodetest.jsonl \
  --models heuristic-small,openai-compatible \
  --strategies direct,plan_then_code \
  --samples-per-task 5 \
  --ks 1,5 \
  --max-tasks 100 \
  --generate-report \
  --output-dir results \
  --stem open_model_baseline
```

### 5) Multi-Model Registry (Bring Your Own APIs)
Copy and edit `configs/models.example.json`, then run:
```bash
export LOCAL_LLM_API_KEY=...
export DEEPSEEK_API_KEY=...

python3 scripts/run_eval.py \
  --dataset data/tinycodetest.jsonl \
  --model-config configs/models.example.json \
  --models qwen-small-local,deepseek-medium-hosted,heuristic-small-alias \
  --strategies direct,plan_then_code \
  --samples-per-task 5 \
  --ks 1,5 \
  --max-tasks 120 \
  --generate-report \
  --output-dir results \
  --stem byo_models
```

### 6) Bring Your Own Dataset + Train Export
Validate your dataset:
```bash
python3 scripts/validate_dataset.py --dataset path/to/your_dataset.jsonl --check-canonical
```

Export train-ready prompt/completion JSONL:
```bash
python3 scripts/export_train_data.py \
  --dataset path/to/your_dataset.jsonl \
  --output data/train/finetune_data.jsonl
```

Generate a dashboard later from any eval JSON:
```bash
python3 scripts/generate_report.py \
  --input results/byo_models.json \
  --output results/byo_models.html
```

### 7) Web App Wrapper (Upload + Pick Models + Run + View)
```bash
python3 scripts/web_app.py --host 127.0.0.1 --port 8787 --auto-generate-dataset
```

Open:
- `https://skill-deploy-c2xu51b9yp.vercel.app/`
- `http://127.0.0.1:8787`

Web UI features:
- upload a dataset JSONL or use an existing path
- upload/select model config JSON and pick models
- choose strategies + eval settings
- run evaluation from the browser
- open JSON/Markdown/HTML artifacts and preview report directly
- optional verification trace output (`.verification.md`) that shows verifier results per task attempt (PASS/FAIL, reward, case counts, first failure reason)
- run metadata is saved in `results/run_meta/` (or `/tmp/...` on Vercel) and artifacts in `results/web_runs/`

Notes:
- Deploy returns a preview URL and claim URL.
- On Vercel, run files are stored in temporary server storage (`/tmp/...`) and may reset.
- If the default dataset path does not exist on Vercel, the app auto-generates a small default dataset in `/tmp`.
- In a run page, click `VERIFY` to inspect the per-attempt verification trace.
- If you want a stable managed domain, claim the deployment URL and redeploy from your claimed project.

## Dataset Design
Default dataset size: 600 tasks
- easy: 200
- medium: 200
- hard: 200
- adversarial-tagged: 478

Task families include arithmetic transforms, string/list manipulation, and harder DP/graph tasks (e.g., edit distance, coin change, shortest path, LIS, word ladder).

## Deterministic Verifier
Verifier behavior:
- isolated subprocess execution (`python -I`)
- per-attempt timeout
- memory cap (best effort via resource limits on Unix)
- strict test equality checks
- reward = weighted fraction of unit tests passed

Each task has fixed test cases, so reward is deterministic for a given completion.

## Scoring
- `pass@k`: standard estimate from `n` sampled completions per task
- `mean_reward`: average fraction of tests passed
- adversarial metrics: same scoring restricted to `adversarial=true` tasks
- bucket metrics: separate scores for easy/medium/hard

## Baseline Snapshot (2026-02-05 UTC, 120-task slice)
Source artifacts:
- `results/baseline_120.json`
- `results/baseline_120.md`

| Rank | Model | Strategy | pass@1 | pass@5 | Mean Reward | Adv pass@1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | reference-oracle | direct | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | reference-oracle | plan_then_code | 1.000 | 1.000 | 1.000 | 1.000 |
| 3 | heuristic-medium | direct | 0.635 | 0.758 | 0.635 | 0.549 |
| 4 | heuristic-medium | plan_then_code | 0.635 | 0.758 | 0.635 | 0.549 |
| 5 | heuristic-small | direct | 0.233 | 0.342 | 0.261 | 0.112 |
| 6 | heuristic-small | plan_then_code | 0.233 | 0.342 | 0.261 | 0.112 |

## Failure Modes
- brittle pattern matching on adversarial inputs
- invalid signatures or missing required function names
- timeouts on harder search/DP tasks
- superficially correct logic that fails edge-case assertions

See `docs/one_pager.md` for a concise benchmark framing.
