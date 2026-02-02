# Phase 3: Multi-step Residual Search

Experiments:
- Horizon 2 (top_k=4, branch_k=2, swap_penalty=0.15, duration_scale=5e-4, progress_weight=0.1). Command: `python3 -m quantum_routing_rl.eval.run_eval --suite pressure --out artifacts ... --residual-multistep --residual-horizon 2`. Summary: `artifacts/summary_noise_unguarded.csv`. Outcomes: overall_log_success_mean for residual_multistep = -0.506 (qiskit_sabre_best = -0.332); swap ratios vs sabre by graph — grid_3x3: 2.36x, heavy_hex_15: 3.57x, ring_8: 5.72x; 701/720 rows flagged for constraint_violation (>1.3x swaps/depth/duration).
- Horizon 3 (same config, `artifacts/summary_noise_unguarded_h3.csv`). Outcomes: overall_log_success_mean = -0.426 (still below sabre); swap ratios — grid_3x3: 2.10x, heavy_hex_15: 2.48x, ring_8: 2.90x; 698/720 rows violate the 1.3x constraints.

Conclusion: Single- and multi-step residual search with learned costs does not outperform SABRE under realistic constraints on these benchmarks.
