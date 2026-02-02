# Phase 3: Multi-step Residual Search

Experiments:
- Horizon 2 (top_k=4, branch_k=2, swap_penalty=0.15, duration_scale=5e-4, progress_weight=0.1). Command: `python3 -m quantum_routing_rl.eval.run_eval --suite pressure --out artifacts ... --residual-multistep --residual-horizon 2`. Summary: `artifacts/summary_noise_unguarded.csv`. Outcomes: overall_log_success_mean for residual_multistep = -0.506 (qiskit_sabre_best = -0.332); swap ratios vs sabre by graph — grid_3x3: 2.36x, heavy_hex_15: 3.57x, ring_8: 5.72x; 701/720 rows flagged for constraint_violation (>1.3x swaps/depth/duration).
- Horizon 3 (same config, `artifacts/summary_noise_unguarded_h3.csv`). Outcomes: overall_log_success_mean = -0.426 (still below sabre); swap ratios — grid_3x3: 2.10x, heavy_hex_15: 2.48x, ring_8: 2.90x; 698/720 rows violate the 1.3x constraints.

Conclusion: Single- and multi-step residual search with learned costs does not outperform SABRE under realistic constraints on these benchmarks.

## Phase 4: Weighted / Noise-Aware SABRE

Implementation highlights:
- Weighted distance cache (`weighted_distance.py`) combines calibrated error, duration/T2, and crosstalk with a stability floor; router runs 8 stochastic trials with SabreLayout-derived seeds and picks the best (overall_log_success, then swaps, depth with a small tie tolerance).
- Router scoring/decay mirrors the teacher and keeps the tightened SABRE candidate set; snapshot handling averages the first two drift snapshots.

Validation sweep (pressure subset, `pressure_qasm=3`, `hardware_samples=2`, seed=13):
- Grid over alpha_time {0.0, 0.2, 0.5, 1.0} × beta_xtalk {0.0, 0.2, 0.5} × snapshot_mode {avg, bucket} × trials {4, 8}.
- All passing configs and scores recorded in `artifacts/sweep/sweep_configs.json`; performance clustered, so we carried forward three representatives: (0, 0, avg, 8), (0, 0.2, bucket, 8), and the noise-aware (0.5, 0.2, avg, 8).

Full noise-aware evaluation (pressure suite, seeds 13/17/23, 10 hardware draws with drift, directional errors, crosstalk):
- Files: `artifacts/results_noise_unguarded_weighted*.csv`, `summary_noise_unguarded_weighted*.csv`, `summary_noise_unguarded_weighted*.csv`, plots in `artifacts/plots_noise_unguarded_weighted/weighted_vs_qiskit_overall_success.{png,pdf}`.
- Representative config (alpha_time=0.5, beta_xtalk=0.2, snapshot_mode=avg, trials=8, frontier_size=4) beats the qiskit_sabre_best reference on overall_log_success while staying within 1.3× on swaps/twoq_depth/total_duration for every pressure graph:
  - grid_3x3: overall_log_success_mean −0.249 ± 0.335 vs qiskit −0.301; swap ratio 1.18×; depth ratio 1.01×; duration ratio 0.93×.
  - heavy_hex_15: overall_log_success_mean −0.390 ± 0.505 vs qiskit −0.410; swap ratio 1.17×; depth ratio 0.96×; duration ratio 0.90×.
  - ring_8: overall_log_success_mean −0.264 ± 0.483 vs qiskit −0.286; swap ratio 1.12×; depth ratio 0.99×; duration ratio 0.95×.
- The other two swept configs matched these metrics (within rounding), indicating robustness to small changes in alpha/beta/snapshot_mode under the current distance floor.

Bottom line: the WeightedSABRE variant delivers higher noise-aware success than qiskit_sabre_best on all pressure graphs while honoring the ≤1.3× swap/depth/duration constraint.
