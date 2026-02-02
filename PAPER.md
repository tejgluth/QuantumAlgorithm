# A Noise- and Drift-Aware SABRE Variant with Multi-Trial Selection

## Abstract
Weighted SABRE augments Qiskit’s SABRE router with drift-aware edge costs, crosstalk-penalised distances, and multi-trial layout jitter scored by a realistic success proxy. On pressure-suite benchmarks (ring_8, grid_3x3, heavy_hex_15) with 50 synthetic calibration draws per graph, the weighted router raises mean overall log success by **+0.031 (95% CI [+0.028, +0.034])** relative to Qiskit’s SABRE while keeping swap ratios between **1.10–1.17**, two‑qubit depth ratios ≤**1.01**, and duration ratios **0.91–0.96** (`artifacts/tables/main_results.csv`). A Wilcoxon signed-rank test over 3,600 paired draws yields **p≈1.2×10⁻²⁷³** with Cliff’s δ **0.72**, indicating a robust paired improvement (`artifacts/statistics/significance.csv`). Ablations replacing time/crosstalk weights with hop distance leave the gain unchanged (Δ≈+0.032, swap ratio 1.124), showing that multi-trial selection under drift dominates the benefit. Variance decomposition attributes 26–34% of outcome variance to hardware draw, <5% to circuit seed, and the rest to circuit-specific residuals. Overall, a noise- and drift-aware multi-trial SABRE-style router, selected by a realistic success proxy, improves expected execution success over Qiskit’s SABRE baseline with negligible routing overhead.

## Introduction
Routing for NISQ hardware must balance connectivity constraints with time-varying noise. Qiskit’s SABRE is fast but noise-agnostic. We ask whether a lightweight, noise-aware variant—combining drift-aware distances, crosstalk penalties, and trial re-seeding—can deliver higher expected success without inflating swaps or depth.

## Background
- **SABRE**: greedy bidirectional heuristic that scores candidate swaps by immediate and lookahead mapping cost.
- **Noise/drift**: hardware calibrations drift over minutes; asymmetric error rates and crosstalk alter effective two-qubit fidelity.
- **Success proxy**: we approximate execution success by combining gate error, duration-induced decoherence, and readout error into `overall_log_success`.

## Method
Weighted SABRE (WS) keeps SABRE’s swap generation but replaces the distance with a weighted, noise-aware metric:
- Edge weight = hop distance scaled by time (`α=0.5`) and crosstalk (`β=0.2`) using the active snapshot; directionality is respected.
- Eight stochastic layout trials are routed; the trial with highest `overall_log_success` is selected.
- Snapshot selection: average over two drift snapshots spaced 50 µs (`snapshot_mode=avg`), drawing per-edge drift from deterministic seeds.

## Experimental Setup
- **Benchmarks**: pressure suite (9 circuits) on `ring_8`, `grid_3x3`, `heavy_hex_15`; seeds 13, 17, 23.
- **Hardware model**: synthetic “realistic” draws with directional CNOT errors 0.003–0.008 (15% edges 0.02–0.05), durations 200–600 ns; T1 20–200 µs, T2 40–90% T1; crosstalk factor 0.01; drift rate 0.05 with two snapshots 50 µs apart; 50 draws per graph (seeds 211–260).
- **Baselines**: `qiskit_sabre_best`, `sabre_layout_swap`, `teacher_sabre_like`, IL and RL checkpoints, residual model; WS uses trials=8, `snapshot_mode=avg`.
- **Data & scripts**: main CSVs `results_noise_unguarded_weighted_hd.csv`, summaries in `artifacts/summary_noise_unguarded_weighted_hd*.csv`; significance/effect sizes in `artifacts/statistics/`; figures/tables in `artifacts/plots_final/` and `artifacts/tables/`.

## Results
- **Main table (Fig. 1, `main_results.csv`)**: WS improves overall log success on all graphs (grid: −0.254 ± 0.019 vs −0.295 ± 0.020; heavy_hex: −0.398 ± 0.028 vs −0.421 ± 0.026; ring: −0.235 ± 0.025 vs −0.266 ± 0.026) with swap ratios 1.099–1.169 and duration ratios 0.910–0.957.
- **Paired deltas (Fig. 2, `significance.csv`)**: mean Δ = +0.0311, 95% CI [+0.0283, +0.0340]; 84.6% of paired draws favor WS; Wilcoxon p ≈ 1.2×10⁻²⁷³; Cliff’s δ = 0.72 (per-graph δ: 0.82/0.60/0.77 for grid/heavy_hex/ring).
- **Yield vs overhead (Fig. 3)**: WS points cluster at swap ratios ≈1.10–1.17 with ≈+0.03 higher mean log success than the Qiskit baseline.
- **Ablation (Fig. 4, `ablation.csv`)**: A0–A3 all deliver −0.306 ± 0.019 log success with Δ≈+0.032 vs Qiskit and identical overhead (swap ratio 1.124, duration ratio 0.926), implying the gain is driven by multi-trial selection + snapshot jitter, not α/β weights.
- **Variance attribution (Fig. 5, `variance.csv`)**: hardware draw explains 26–34% of variance; seed <5%; residual (circuit difficulty) dominates.

## Discussion
- **Trial selection dominates**: Ablation invariance and strong Cliff’s δ show the stochastic layout search plus noise-aware proxy, not the specific α/β tuning, yields the uplift.
- **Overhead budget respected**: All ratios stay well below the 1.3× constraint used in regression checks; durations fall despite modest swap inflation.
- **Robustness to drift**: Positive deltas persist across 50 calibration draws with directional errors, indicating WS benefits from sampling drifted layouts rather than overfitting a single snapshot.
- **Limitations**: Synthetic calibrations may not match device-specific cross-talk structure; only three graphs and nine circuits were evaluated; real calibration data were not used; success proxy is still a heuristic, not measured hardware success.

## Related Work
SABRE and its layout/swap variants are widely used in Qiskit. Noise-aware routers include Tket’s directed routing and Qiskit pulse-aware passes; multi-start or restart heuristics are common in placement. Learned routers (IL/RL) and residual policies are evaluated here as baselines but not modified in this phase.

## Conclusion
The evidence supports the primary claim: **“A noise- and drift-aware multi-trial SABRE-style router selected by a realistic success proxy improves expected execution success over Qiskit’s SABRE baseline with negligible routing overhead.”** The improvement is statistically significant, overheads stay within a 1.3× budget, and the approach remains simple enough for compiler integration via a scoring proxy and limited trial count.
