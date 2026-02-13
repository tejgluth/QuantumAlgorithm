# Reproducibility Checklist

This project is designed to regenerate the paper-ready artifacts deterministically on a single host. Run `python3 scripts/bootstrap.py --dev` once (or `py -3 scripts\\bootstrap.py --dev` on Windows) and keep using the same interpreter for `scripts/run.py` commands.

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

## Commands to regenerate the gauntlet + verdict story
Recommended flow (CPU or GPU):

```bash
HARDWARE_DRAWS=50 python3 scripts/run.py gauntlet-full
python3 scripts/run.py invariants
python3 scripts/run.py validate-proxy-extended --include-weighted
python3 scripts/run.py verdict
```

- Gauntlet results land under `artifacts/gauntlet/<timestamp>/` with `results_gauntlet_full.csv` and `summary_gauntlet_full.csv`.
- Invariants are written next to the results in `invariants/`.
- Proxy validation artifacts are placed in `proxy_validation_extended/` under the same gauntlet run.
- `FINAL_VERDICT.md` is written inside the gauntlet run and copied to `artifacts/FINAL_VERDICT.md`.

For legacy paper figures/tables, `make reproduce-paper` still works on systems with `make` available; it reuses the same seeds and noise settings but remains optional to the main gauntlet storyline.

## Version Pinning (recorded in `artifacts/metadata.json`)
- Python: 3.12.10
- Qiskit: 2.3.0
- PyTorch: 2.10.0
- OS/Platform: macOS-26.2-arm64
- Device: MPS available (CUDA not available)

All commands and seeds above are reflected in the checked-in artifacts; rerunning with the same parameters should reproduce the numbers used in PAPER.md exactly.
