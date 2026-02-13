# Contributing

Thanks for helping improve `quantum-routing-rl`! Please keep the scope focused on routing quality, evaluation, and reproducibility.

## Setup
- `python3 scripts/bootstrap.py --dev` (uses `.venv` by default; override with `QRRL_VENV_DIR`)
- Style and tests: `python3 scripts/run.py lint` then `python3 scripts/run.py test`

## Workflow
- Prefer small, test-backed changes. Avoid regenerating large artifacts unless necessary.
- Prefer the Python runner: `python3 scripts/run.py gauntlet-small` or `gauntlet-full`; Make remains optional.
- New routing baselines should integrate through `src/quantum_routing_rl/eval/run_eval.py` and log metrics via `_result_record`.
- When adding benchmarks, document the source dataset and update `REPRODUCIBILITY.md` with seeds and commands.

## Scope notes
- Weighted SABRE and fair SABRE baselines are the primary focus. Legacy IL/RL training scripts live in `experiments/legacy_rl/` and are not part of the default install path.
- Do not commit secrets or environment-specific credentials. `.env` should stay ignored.

## Opening a PR
- Ensure a clean `git status` and that `make lint && make test` pass.
- Describe the experiment settings if results are added; never fabricate metrics.
- Agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
