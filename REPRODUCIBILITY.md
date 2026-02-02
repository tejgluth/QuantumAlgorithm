# Reproducibility Checklist

This project is designed to regenerate the paper-ready artifacts deterministically on a single host. All commands assume the repository root and a populated `.venv` (run `make setup` once).

## Hardware Model and Noise Assumptions
- **Coupling graphs:** `ring_8`, `grid_3x3`, `heavy_hex_15`.
- **Synthetic calibration draws:** `HardwareModel.synthetic` with `profile=realistic`, `directional_mode=True`, `drift_rate=0.05`, `snapshot_spacing_ns=50_000`, `snapshots=2`, `crosstalk_factor=0.01`.
- **Per-edge parameters:** typical CNOT error sampled in `[0.003, 0.008]`; ~15% “bad” edges in `[0.02, 0.05]`; durations in `[200, 600]` ns; directional scale factors in `[0.8, 1.2]`.
- **Per-qubit parameters:** `t1` in `[20 000, 200 000]` ns, `t2` = `t1·U[0.4, 0.9]`, single‑qubit error `U[1e-4, 5e-3]`, readout error `U[0.01, 0.05]`, single‑qubit duration `U[20, 80]` ns.

## Seed Handling and Determinism
- **Circuit/layout seeds:** `--seeds 13 17 23` (shared by all baselines).
- **Pressure-suite synthesis:** `--pressure-seed 99`, `--pressure-qasm 6`.
- **Hardware draws:** 50 samples per graph via `--hardware-seed-base 211`, producing seeds 211–260.
- **Bootstrap / stats:** paired-delta CIs use `--seed 1234`; Wilcoxon uses the same seed for the bootstrap component.
- **Ablations:** same circuit seeds, `--hardware-draws 30`, snapshot_mode=`avg`, trials=`8`; skip if results already exist to preserve prior artifacts.

## Hardware Draw Generation
Calibration snapshots are generated on the fly inside `run_eval` by calling `HardwareModel.synthetic(graph, seed=hardware_seed, ...)` for each hardware seed and graph. Drifted snapshot `t1/t2` and CNOT parameters are produced by the deterministic `_drift_*` helpers using `seed + snapshot_index`, so the full draw set is reproducible from the seed tuple `(graph_id, hardware_seed)`.

## Commands to Regenerate Figures and Tables
The single entry point is:

```bash
make reproduce-paper
```

This performs (skipping reruns when outputs already exist):
- Weighted SABRE pressure evaluation with noise/drift (`results_noise_unguarded_weighted_hd.csv`, `summary_noise_unguarded_weighted_hd.csv`).
- Paired deltas + Wilcoxon + effect sizes (`artifacts/deltas/`, `artifacts/statistics/`).
- Variance breakdown (`artifacts/variance/variance_breakdown.csv`).
- Ablation sweep A0–A3 (`artifacts/summary_ablation*.csv`).
- Yield vs overhead points (`artifacts/plots_yield_overhead/`).
- Paper-ready tables (`artifacts/tables/`) and plots (`artifacts/plots_final/Fig*.png|pdf`).

Expected runtime: dominated by the 50-draw pressure evaluation (~25k routed circuits; per-circuit routing runtime averages <3 ms in `summary_noise_unguarded_weighted_hd.csv`). On an Apple silicon laptop this completes within a few minutes; budget ~10 minutes to be safe.

### Outputs by Artifact
- **Main results table & figure:** `artifacts/tables/main_results.{csv,tex}`, `artifacts/plots_final/Fig1_MainResults.{png,pdf}`.
- **Paired delta histogram + stats:** `artifacts/plots_final/Fig2_PairedDeltas.{png,pdf}`, `artifacts/statistics/significance.csv`, `effect_size.csv`.
- **Yield vs overhead:** `artifacts/plots_final/Fig3_YieldVsOverhead.{png,pdf}`.
- **Ablation:** `artifacts/tables/ablation.{csv,tex}`, `artifacts/plots_final/Fig4_Ablation.{png,pdf}`.
- **Variance attribution:** `artifacts/tables/variance.{csv,tex}`, `artifacts/plots_final/Fig5_VarianceBreakdown.{png,pdf}`.

## Version Pinning (recorded in `artifacts/metadata.json`)
- Python: 3.12.10
- Qiskit: 2.3.0
- PyTorch: 2.10.0
- OS/Platform: macOS-26.2-arm64
- Device: MPS available (CUDA not available)

All commands and seeds above are reflected in the checked-in artifacts; rerunning with the same parameters should reproduce the numbers used in PAPER.md exactly.
