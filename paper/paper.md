# A Noise- and Drift-Aware Multi-Trial SABRE Router

## Abstract
Selecting among multiple SABRE routing trials using a drift-aware success proxy improves expected execution success over Qiskit’s SABRE baseline while keeping overhead negligible. On pressure-suite benchmarks (ring_8, grid_3x3, heavy_hex_15) with 50 synthetic calibration draws per graph, the weighted SABRE variant lifts overall log success by **+0.031 (95% CI [+0.028, +0.034])** relative to Qiskit’s SABRE, with swap ratios **1.10–1.17**, two‑qubit depth ratios ≤**1.01**, and duration ratios **0.91–0.96** (`tables/main_results.csv`). A paired Wilcoxon test over 3,600 draws gives **p < 1e‑10** and Cliff’s δ **0.72 (large)** (`tables/significance_effect.csv`). Ablations that remove time/crosstalk weights preserve the gain, indicating the multi-trial selection loop—not the local edge weights—is the primary mechanism.

## 1 Introduction
Near-term devices suffer drift, directionality, and crosstalk. Standard SABRE routing [[@li2019sabre]] is fast but noise-agnostic; Qiskit’s implementation keeps a single trial. We ask whether a lightweight, noise-aware variant that (1) jitters initial layouts, (2) routes multiple trials, and (3) selects by a hardware-informed success proxy can raise expected yield without inflating swaps or depth. The primary claim: **objective-aware selection among multiple SABRE trials yields statistically significant improvements in expected execution success with negligible overhead.**

## 2 Methods
- **Weighted SABRE:** SABRE swap generation with edge weights combining hop distance, duration (α = 0.5), and crosstalk penalty (β = 0.2) using the active hardware snapshot.
- **Success proxy selection:** Each trial is scored by `overall_log_success`, which aggregates two‑qubit error, decoherence from gate durations, and readout error on a drift-aware schedule.
- **Fair SABRE control:** A new baseline `qiskit_sabre_trialsN` runs Qiskit’s SABRE N times with distinct seeds and selects the route that minimises swaps, then depth, then duration. Default N = 8 matches the weighted SABRE trial budget.
- **Ablations:** A0–A3 toggle α/β weights and snapshot mode while keeping the multi-trial selection.

## 3 Experimental Setup
- **Benchmarks:** Pressure suite (9 QASMBench circuits) on coupling graphs `ring_8`, `grid_3x3`, `heavy_hex_15`; seeds 13, 17, 23.
- **Hardware model:** Synthetic “realistic” draws with directional CNOT errors 0.003–0.008 (15% edges 0.02–0.05), durations 200–600 ns, T1 20–200 µs, T2 40–90% T1, crosstalk factor 0.01, drift rate 0.05 with two snapshots 50 µs apart; 50 draws per graph (seeds 211–260).
- **Baselines:** `sabre_layout_swap`, `qiskit_sabre_best`, `qiskit_sabre_trials8`, `teacher_sabre_like`, IL/RL checkpoints, residual search; weighted SABRE uses trials=8 and `snapshot_mode=avg`.
- **Metrics:** swaps inserted, two‑qubit depth, total duration, and noise-aware success proxies (`overall_log_success`, `log_success_proxy`, duration proxy).

## 4 Results
- **Overall performance (Fig. 1, `tables/main_results.csv`):** Weighted SABRE improves mean overall log success on every graph while keeping swap and duration ratios within a 1.3× budget.
- **Significance and effect size (Fig. 2, `tables/significance_effect.csv`):** mean paired Δ = +0.0311 with 95% CI [+0.0283, +0.0340]; **p < 1e‑10**; Cliff’s δ = 0.72 (large). Per-graph δ ranges 0.60–0.82, all “medium” or better.
- **Yield vs overhead (Fig. 3):** Weighted points cluster near swap ratios 1.10–1.17 with ≈+0.03 log-success uplift over Qiskit SABRE.
- **Ablation (Fig. 4, `tables/ablation.csv`):** Removing α/β weighting leaves Δ≈+0.032 and swap ratio ≈1.12, reinforcing that trial selection + proxy scoring drives the gain.
- **Variance (Fig. 5, `tables/variance.csv`):** Hardware draw explains 26–34% of variance; seed <5%; remaining variance reflects circuit difficulty.

## 5 Mechanism
Ablations show negligible change when α/β are zero, so **the benefit comes from multi-trial layout exploration plus success-proxy selection**. This is a compiler design pattern: generate a small set of valid routes under heterogeneous hardware, then pick the route that maximises the execution objective (expected success) instead of a purely structural surrogate (swap/depth). The proxy-aware selector is cheap (eight trials) yet captures drift, directionality, and crosstalk ignored by standard SABRE scoring.

## 6 Limitations and Scope
Synthetic-but-structured hardware draws allow controlled drift and crosstalk studies, but they do not substitute for live calibration. Replicating on public calibration snapshots would require: (1) ingesting calibration JSON for multiple dates, (2) mapping gate errors, durations, T1/T2, and readout into `HardwareModel` fields, and (3) repeating the paired evaluation across snapshots to capture day-to-day drift. We report expected-yield proxies, not hardware execution rates, and only three coupling graphs plus nine circuits are covered.

## 7 Reproducibility
- `make reproduce-paper` regenerates results, tables, figures, and `paper/figures`/`paper/tables` from raw evaluation runs (50 hardware draws, trials=8).
- Intermediate commands: `make lint`, `make test`, `SKIP_RL=1 make eval-noise-unguarded-weighted HARDWARE_DRAWS=50`, and `python -m quantum_routing_rl.eval.paper_assets ...` (see `Makefile`).
- All figures live in `paper/figures/` (symlinked to `artifacts/plots_final/`); tables are copied to `paper/tables/`.

## 8 Conclusion
Noise- and drift-aware trial selection atop SABRE yields a statistically significant, large-effect improvement in expected execution success with negligible overhead. Matching Qiskit’s trial budget shows the gain is not just “more shots” but the combination of noise-aware scoring and objective-driven selection, making the approach practical as a compiler pass.

## References
References are provided in `references.bib` (minimal set: SABRE, Qiskit, noise-aware mapping).
