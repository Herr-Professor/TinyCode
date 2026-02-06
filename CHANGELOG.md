# Changelog

## [0.2.0] - 2026-02-06

### Added
- Optional Verifiers adapter: `tinycodetest.verifiers_adapter`
- Bootstrap confidence intervals for pass@k via `--confidence-intervals`
- Verification trace export utility: `scripts/export_verification_trace.py`
- Dataset uniqueness validation: `validate_dataset_uniqueness()`
- Model config validation with descriptive errors
- Environment-aware sandbox defaults: `SandboxConfig.for_environment()`
- Web UI support for confidence interval runs
- Smoke test utility: `scripts/smoke_test.py`

### Changed
- Consolidated timeout/memory defaults to a single source of truth
- Improved model configuration diagnostics and runtime warnings
- Expanded benchmark docs for confidence interval interpretation

### Fixed
- Web app default dataset behavior is environment-aware (local vs Vercel)
- Default dataset generation can be overridden with `TCT_DEFAULT_TASKS`
