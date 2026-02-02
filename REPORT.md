# Phase 4: Weighted SABRE Validation (Noise + Drift)

## Problem
Quantify whether WeightedSABRE improves predicted execution success under realistic, drift-aware hardware models without exceeding the constraint budget (swaps/twoq_depth/total_duration ≤ 1.3× `qiskit_sabre_best`).

## Experimental Setup
- Benchmarks: pressure suite on `ring_8`, `grid_3x3`, `heavy_hex_15` (9 circuits).
- Seeds: 13, 17, 23. Hardware draws per seed: **50**; directional errors, crosstalk=0.01, drift rate=0.05, two snapshots spaced 50 µs.
- Config: `alpha=0.5`, `beta=0.2`, `snapshot_mode=avg`, `trials=8`, frontier=4. Command: `SKIP_RL=1 make eval-noise-unguarded-weighted HARDWARE_DRAWS=50`.
- Key artifacts: `artifacts/results_noise_unguarded_weighted_hd.csv`, `summary_noise_unguarded_weighted_hd.csv`, `summary_noise_unguarded_weighted_hd_std.csv`, plots in `artifacts/plots_noise_unguarded_weighted/`, metadata in `artifacts/metadata.json`.

## Main Results (95% CI of mean overall_log_success)
| graph | qiskit_sabre_best | weighted_sabre | swap ratio | depth ratio | duration ratio |
| --- | --- | --- | --- | --- | --- |
| grid_3x3 | −0.295 ± 0.020 | **−0.254 ± 0.019** | 1.17 | 1.01 | 0.92 |
| heavy_hex_15 | −0.421 ± 0.026 | **−0.398 ± 0.028** | 1.14 | 0.97 | 0.91 |
| ring_8 | −0.266 ± 0.026 | **−0.235 ± 0.025** | 1.10 | 0.99 | 0.96 |

All overhead ratios remain <1.2×, comfortably inside the 1.3× constraint ceiling.

## Paired Delta Analysis
- Dataset: 3,600 paired draws (circuit, graph, seed, hardware_seed) saved in `artifacts/deltas/paired_deltas.csv`.
- Mean delta_success = **+0.0311** with 95% bootstrap CI [0.0283, 0.0340]; **84.6%** of pairs favor WeightedSABRE (`artifacts/deltas/paired_delta_summary.csv`, plots in `artifacts/plots_paired_deltas/`).
- Per-graph positive fractions: grid_3x3 91.1%, heavy_hex_15 78.2%, ring_8 85.4%.

## Variance Attribution (centered per circuit)
- Hardware heterogeneity explains ~30% of variance in overall_log_success; seed-to-seed randomness explains <1% on average; the remainder is circuit-specific/residual (`artifacts/variance/variance_breakdown.csv`, plots in `artifacts/variance/`).
- Example (weighted_sabre): grid_3x3 hardware 33%, seed 0.6%, residual 67%.

## Ablation Study (30 hardware draws, trials=8, snapshot_mode=avg)
- Configs: A0 (α=0, β=0), A1 (α>0, β=0), A2 (α=0, β>0), A3 (α>0, β>0).
- Outcomes (`artifacts/summary_ablation.csv`, `artifacts/plots_ablation/`): all four ablations produce nearly identical success (grid −0.259, heavy_hex −0.389, ring −0.253) and overhead (swap ratios 1.09–1.17, duration ratios 0.89–0.96). The gain therefore stems mainly from layout jitter + multi-trial search, not from the specific α/β weights.

## Yield vs Routing Overhead
- Scatter (`artifacts/plots_yield_overhead/yield_vs_overhead.png`): WeightedSABRE points sit at swap ratios 1.10–1.17 with higher success (≈+0.03) than the qiskit baseline at ratio 1.0 (`yield_vs_overhead_points.csv`).

## Limitations
- Residual variance is large (circuit difficulty dominates); hardware models are synthetic with assumed drift/crosstalk parameters.
- Ablation insensitivity suggests the current α/β weighting has limited leverage; benefits may mainly come from stochastic restarts rather than noise-aware distance terms.
- Only pressure-suite graphs and three seeds were tested; no cross-validation on larger layouts or real calibration data.

## Takeaway for Compiler Developers
WeightedSABRE (α=0.5, β=0.2, trials=8) delivers a small but statistically robust improvement in predicted execution success under drift-aware noise models, while keeping swap/depth/duration overhead below 1.2× relative to `qiskit_sabre_best`. For hardware with noticeable drift or directional errors, enabling this routing variant is a low-risk knob: expect ~3e−2 uplift in log-success in ~85% of cases with minor swap inflation and shorter predicted duration.
