# quantum-routing-rl

Learned qubit routing vs SABRE-family baselines in Qiskit.

Quickstart:
  make setup
  make test
  make lint
  make eval-small

Reproducible runs:
- `make eval-teacher` (parity guard on pressure graphs)
- `make eval-pressure` and `make eval-noise` (includes il_soft, rl_ppo, SABRE baselines; swap-guard checks enforced)
- `make eval-full` (multi-seed full suite, writes `artifacts/results.csv` / `summary.csv` and plots)

Full run (later):
  make eval-full

Codex reads AGENTS.md automatically.
Main instructions are in prompts/START_HERE.md.

Suites:
- `make eval-small` → dev sanity suite (writes `artifacts/results_dev.csv` and `artifacts/plots_dev/`).
- `make eval-pressure` → synthetic + selected QASMBench circuits on pressure graphs (ring_8, grid_3x3, heavy_hex_15).
- `make eval-full` → full QASMBench + pressure circuits, multi-seed; produces merged `artifacts/results.csv`, `artifacts/summary.csv`, and plots in `artifacts/plots/`.

New routing-pressure generators live in `src/quantum_routing_rl/benchmarks/synthetic_generator.py`.
The learned router is available as a Qiskit `TransformationPass` via
`quantum_routing_rl.qiskit_pass.learned_swap_router.LearnedSwapRouter`.
