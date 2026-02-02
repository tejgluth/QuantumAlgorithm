# quantum-routing-rl

Weighted SABRE and learned routing baselines for Qiskit, evaluated on drift- and noise-aware hardware models.

## Quick takeaways (reviewer-safe)
- Weighted SABRE lifts mean overall log success by **+0.031** (95% CI [+0.028, +0.034]) vs Qiskit SABRE across 3,600 paired draws; Wilcoxon **p < 1e‑10**, Cliff’s δ **0.72 (large)**.
- Overhead stays small: swap ratio **1.10–1.17**, two‑qubit depth ratio ≤**1.01**, duration ratio **0.91–0.96** (Fig1 / `artifacts/tables/main_results.csv`).
- Fairness check: the new `qiskit_sabre_trials8` baseline matches Weighted SABRE’s trial budget; results are logged alongside `weighted_sabre` in `artifacts/results_noise_unguarded_weighted_hd.csv`.

## Why this matters (industry)
- Objective-aware selection: pick the routed trial with the best success proxy instead of the first layout found.
- Drift/crosstalk robustness: 26–34% of variance comes from calibration draw (`artifacts/tables/variance.csv`); multi-trial selection hedges against hardware drift.
- Negligible cost: eight trials add milliseconds yet preserve swap/depth budgets enforced by regression checks (≤1.3× Qiskit SABRE).

### 10-line drop-in integration (Qiskit)
```python
from qiskit.transpiler import CouplingMap
from quantum_routing_rl import route_with_weighted_sabre, WeightedDistanceParams, make_synthetic_hardware
cmap = CouplingMap([[0,1],[1,2],[2,3],[3,0]])
hardware = make_synthetic_hardware(cmap.graph, seed=211, drift_rate=0.05, snapshots=2, crosstalk_factor=0.01)
params = WeightedDistanceParams(alpha_time=0.5, beta_xtalk=0.2)
result = route_with_weighted_sabre(circuit, cmap, hardware_model=hardware, trials=8, distance_params=params)
best_circuit = result.circuit
log_success = result.metrics.overall_log_success
```

## Key results (pressure suite, 50 hardware draws)
| Graph        | Δ overall_log_success | Swap ratio | Duration ratio |
|--------------|-----------------------|------------|----------------|
| grid_3x3     | +0.041 [0.038, 0.045] | 1.169      | 0.917          |
| heavy_hex_15 | +0.022 [0.016, 0.029] | 1.141      | 0.910          |
| ring_8       | +0.031 [0.028, 0.034] | 1.099      | 0.957          |
Sources: `artifacts/tables/main_results.csv`, `artifacts/tables/significance_effect.csv` (exact p-values kept there; main text reports p < 1e‑10).

## Quickstart
- Install: `pip install -e .` (optional extras: `pip install -e .[dev]` for lint/tests, `pip install -e .[paper]` for figure generation).
- Examples: `python examples/run_weighted_sabre_on_qasm.py` or `python examples/compare_qiskit_vs_weighted.py` (<2 minutes each).
- Minimal API: `route_with_weighted_sabre`, `run_eval`, `make_synthetic_hardware` exported from `quantum_routing_rl`.

## Reproduce paper artifacts
1. `make lint` && `make test`
2. `SKIP_RL=1 make eval-noise-unguarded-weighted HARDWARE_DRAWS=50` (runs weighted SABRE + fair Qiskit multi-trial baseline)
3. `make reproduce-paper` (paired deltas, variance, tables, figures; syncs `paper/figures` and `paper/tables`)

## Repository guide and scope
- `src/quantum_routing_rl/models/weighted_sabre.py`: multi-trial weighted SABRE implementation.
- `src/quantum_routing_rl/eval/run_eval.py`: evaluation harness and metadata capture.
- `src/quantum_routing_rl/eval/paper_assets.py`: final tables/figures and version pinning.
- `paper/`: publication-grade markdown + figures/tables.
- Legacy RL/IL/residual training scripts are quarantined in `experiments/legacy_rl/` and are **not part of the main paper artifact**.

## License & citation
- License: Apache-2.0 (see `LICENSE`).
- Cite using `CITATION.cff`; references are listed in `paper/references.bib`.
