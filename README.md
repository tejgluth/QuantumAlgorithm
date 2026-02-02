# quantum-routing-rl

Weighted SABRE and learned routing baselines for Qiskit, evaluated on drift- and noise-aware hardware models.

## What this project shows (in 30 seconds)
- Weighted SABRE improves mean overall log success by **+0.031** (95% CI [+0.028, +0.034]) over Qiskit’s SABRE across 3,600 paired draws with **p≈1.2×10⁻²⁷³** and Cliff’s δ **0.72**.
- Overhead stays small: swap ratio **1.10–1.17**, two‑qubit depth ratio ≤**1.01**, duration ratio **0.91–0.96** (Fig1 / `artifacts/tables/main_results.csv`).
- Gains persist across 50 calibration draws with drift/crosstalk and across A0–A3 ablations (Δ≈+0.032 with identical overhead), so multi-trial selection under drift drives the benefit.

## Why this matters for real devices
- Drift and directionality matter: 26–34% of variance in success comes from calibration draw (`artifacts/tables/variance.csv`), and Weighted SABRE’s multi-trial search hedges against it.
- The success proxy blends error, duration, and decoherence; selecting the best of a handful of trials costs milliseconds but recovers yield.
- Constraints stay intact: regression checks enforce swap/depth/duration ≤1.3× Qiskit SABRE, and Weighted SABRE stays well inside that budget.

## How to plug Weighted SABRE into a compiler workflow
1. Install: `make setup` (creates `.venv` with qiskit>=1.0, torch, scipy, etc.).
2. Draw a hardware model (directional, drift-aware):
   ```python
   from qiskit.transpiler import CouplingMap
   from quantum_routing_rl.hardware.model import HardwareModel
   cmap = CouplingMap.from_grid(3, 3)
   hw = HardwareModel.synthetic(cmap.graph, seed=211, profile="realistic",
                                directional=True, drift_rate=0.05,
                                snapshots=2, snapshot_spacing_ns=50_000,
                                crosstalk_factor=0.01)
   ```
3. Route with multi-trial Weighted SABRE and keep the best success-proxy trial:
   ```python
   from quantum_routing_rl.models.weighted_sabre import (
       route_with_weighted_sabre, WeightedSabreWeights
   )
   from quantum_routing_rl.routing.weighted_distance import WeightedDistanceParams
   weights = WeightedSabreWeights()  # SABRE-style decay/lookahead defaults
   distance = WeightedDistanceParams(alpha_time=0.5, beta_xtalk=0.2)
   result = route_with_weighted_sabre(
       circuit, cmap, hardware_model=hw, trials=8,
       router_weights=weights, distance_params=distance,
       snapshot_mode="avg", seed=13,
   )
   routed_circuit = result.circuit
   metrics = result.metrics  # swaps, twoq_depth, overall_log_success, etc.
   ```
4. Drop this into a transpiler pass by wrapping `route_with_weighted_sabre` inside your pass or reusing the existing learned-router pass (`quantum_routing_rl.qiskit_pass.learned_swap_router.LearnedSwapRouter`) for IL/RL policies.

**ASCII flow:**
```
[draw calibrations (50 seeds)]
        ↓
[generate 8 layout trials with SABRE init]
        ↓
[score each trial with overall_log_success proxy]
        ↓
[keep best trial]
        ↓
[run SABRE swap routing on that layout]
```

## Engineering insights for hardware/compiler teams
- Weighted SABRE delivers a **statistically significant uplift** while staying well under the 1.3× overhead guardrail—low-risk to enable on drift-prone devices.
- **Trial selection beats weight tuning**: all A0–A3 ablations match (Δ≈+0.032 with identical overhead), so focus on restart + selection, not α/β tweaking.
- **Hardware variation dominates** (26–34% of variance); sampling multiple calibrations matters more than adding circuit seeds, and the full pipeline is reproducible via `make reproduce-paper`.

## Reproduce the paper artifacts
- Run `make reproduce-paper` (uses `SKIP_RL=1`, `HARDWARE_DRAWS=50`, seeds 13/17/23, hardware seeds 211–260).
- Outputs:
  - Tables: `artifacts/tables/main_results.csv`, `ablation.csv`, `variance.csv`.
  - Plots: `artifacts/plots_final/Fig1_MainResults.png` … `Fig5_VarianceBreakdown.png`.
  - Stats: `artifacts/statistics/significance.csv`, `effect_size.csv`.

## Repository guide
- `src/quantum_routing_rl/models/weighted_sabre.py`: multi-trial weighted SABRE implementation.
- `src/quantum_routing_rl/eval/run_eval.py`: evaluation harness and metadata capture.
- `src/quantum_routing_rl/eval/paper_assets.py`: final tables/figures and version pinning.
- `REPRODUCIBILITY.md`: seeds, hardware model, commands.
- `PAPER.md`: paper-style writeup for reviewers and industry readers.
