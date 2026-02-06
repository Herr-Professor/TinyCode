# TinyCodeTest One-Pager

## What It Measures
TinyCodeTest measures whether a model can produce executable Python functions that satisfy deterministic unit tests under sandbox and timeout constraints. The benchmark emphasizes:
- functional correctness (strict unit tests)
- robustness to edge cases (adversarial subset)
- reliability under repeated sampling (pass@k)

## Task Mix
- 600 total tasks by default
- Balanced difficulty buckets: easy / medium / hard
- Adversarial tasks include edge cases like empty inputs, repeated delimiters, unreachable states, and ambiguous structure patterns

## Scoring
- `pass@k`: per-task success probability estimate from `n` samples, averaged over tasks
- `mean_reward`: average fraction of tests passed (partial credit)
- `adversarial pass@k`: pass@k computed only on adversarial tasks

Reward is deterministic because verifier logic is deterministic and uses fixed tests for each task.

## Failure Modes
- Shallow pattern matching that misses corner cases
- Code that compiles but times out on hard tasks
- Overfitting to common formats and failing adversarial variants
- Invalid function signatures or side effects that break harness assumptions

## What Good Looks Like
- Strong separation between easy/medium/hard curves (no bucket collapse)
- High `pass@1` and increasing `pass@5` without reward instability
- Adversarial performance close to standard split (small drop)
- Reproducible results across seeds and reruns

## Interpreting Uncertainty
When enabled with `--confidence-intervals`, pass@k metrics include 95% confidence intervals.

Method:
- Non-parametric bootstrap resampling of per-task pass@k estimates
- 2000 bootstrap samples by default
- Applied to overall, adversarial, and difficulty-split pass@k aggregates

Interpretation:
- narrow intervals (for example width < 0.10): estimate is relatively stable
- wide intervals (for example width > 0.20): increase tasks and/or samples per task
- with very small runs, intervals can be broad even when point estimates look separated

When to use:
- comparing models with small score gaps
- publishing benchmark results
- sensitivity checks across prompts/configs/seeds
