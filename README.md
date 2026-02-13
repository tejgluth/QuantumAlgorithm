# quantum-routing-rl

Noise-aware quantum circuit routing with the **Weighted SABRE** baseline, a reproducible gauntlet harness, and a FINAL_VERDICT pipeline that can run on Ubuntu, macOS, or Windows (best effort on Windows).

## What you get
- Weighted SABRE router with hardware-aware distances and multi-trial selection.
- Fair Qiskit baselines (SABRE best-of-N, preset opt3) plus deterministic invariant checks.
- Gauntlet suites (small/full/industrial) spanning QASMBench and structured hard cases.
- Validation tools: Aer proxy cross-check (`proxy_validation_extended`) and verdict assembly.

## Quick start (all OS)
- Ubuntu / macOS
  - `python3 scripts/bootstrap.py --dev`
  - `python3 scripts/run.py test`
- Windows (PowerShell)
  - `py -3 scripts\bootstrap.py --dev`
  - `py -3 scripts\run.py test`

`bootstrap.py` creates `.venv` by default (override with `QRRL_VENV_DIR`) and installs qiskit **and** qiskit-aer plus dev tools. No sudo required. `scripts/run.py` keeps the same interpreter so you never depend on `bin/` vs `Scripts/` paths.

## Datasets (QASMBench) and strict mode
- Set `QASMBENCH_ROOT=/abs/path/to/QASMBench` before running full/industrial suites.
- Enable strict mode with `QRRL_REQUIRE_QASMBENCH=1` **or** `--require-qasmbench` on gauntlet/proxy commands to forbid the bundled fixtures; industrial/full tiers will fail fast if the real dataset is missing.

## Run the gauntlet (CPU path)
- `HARDWARE_DRAWS=10 python3 scripts/run.py gauntlet-small`
- Results land in `artifacts/gauntlet/<timestamp>/`. The combined CSV is `results_gauntlet_small.csv` with a matching summary.
- Optional checks:
  - `python3 scripts/run.py invariants`
  - `python3 scripts/run.py validate-proxy-extended --include-weighted`
  - `python3 scripts/run.py verdict`

## Ubuntu + CUDA route
1) `python3 scripts/bootstrap.py --dev --cuda --aer-gpu`
2) Export your dataset + strict guard: `export QASMBENCH_ROOT=/abs/path/to/QASMBench QRRL_REQUIRE_QASMBENCH=1`
3) Smoke check (fast): `HARDWARE_DRAWS=2 python3 scripts/run.py gauntlet-small`
4) Generate the CUDA-aware mega command: `python3 scripts/run.py print-mega-command` (auto-picks `HARDWARE_DRAWS=1000` when `nvidia-smi` is present, else 200).

`--cuda` is best-effort: it tries the stock torch, then CUDA wheels (cu121 → cu118) if `nvidia-smi` is present. `--aer-gpu` is attempted only when CUDA torch is detected.

## Reproduce the paper / verdict story
1) Run the gauntlet tier you need (recommended: full):
   - `HARDWARE_DRAWS=50 python3 scripts/run.py gauntlet-full`
2) Run invariants on the latest gauntlet results (auto-detected):
   - `python3 scripts/run.py invariants`
3) Validate the noise proxy against Aer:
   - `python3 scripts/run.py validate-proxy-extended --include-weighted`
4) Assemble the final claim:
   - `python3 scripts/run.py verdict`

Artifacts live under `artifacts/gauntlet/<timestamp>/` with a copy of `FINAL_VERDICT.md` at `artifacts/FINAL_VERDICT.md`. Gauntlet results/summary filenames include the tier (small/full/industrial).

## Troubleshooting
- Missing venv or wrong interpreter: rerun `python3 scripts/bootstrap.py --dev --venv .venv_alt` and call `scripts/run.py` with the same python.
- Aer not installed: `python3 -m pip install -U qiskit-aer` (bootstrap already includes it).
- Ruff missing: rerun bootstrap with `--dev` or `python3 -m pip install -U ruff`.
- Torch shows CPU only: ensure `nvidia-smi` exists, rerun `python3 scripts/bootstrap.py --cuda`, then `python3 scripts/doctor.py` to confirm.
- Windows path quirks: never call `.venv/bin/python`; always invoke `python scripts/run.py ...` or `py -3 scripts\run.py ...`.

## Final one-liner (Ubuntu + CUDA)
Run everything (bootstrap + gauntlet full + industrial + invariants + proxy validation + verdict) in one pasteable line. This is exactly what `python3 scripts/run.py print-mega-command` emits on a CUDA box:

```bash
export QASMBENCH_ROOT=${QASMBENCH_ROOT:?set QASMBENCH_ROOT to full QASMBench dataset} QRRL_REQUIRE_QASMBENCH=1 HARDWARE_DRAWS=1000 PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python3}; python3 scripts/bootstrap.py --dev --cuda --aer-gpu && $PYTHON_BIN scripts/run.py gauntlet-full --hardware-draws $HARDWARE_DRAWS --hardware-snapshots 3 --hardware-drift 0.01 --hardware-crosstalk 0.02 --hardware-snapshot-spacing 75000.0 --hardware-directional --seeds 13 17 23 29 31 37 41 --qasm-root "$QASMBENCH_ROOT" --selection-seed 11 --require-qasmbench && $PYTHON_BIN scripts/run.py gauntlet-industrial --hardware-draws $HARDWARE_DRAWS --hardware-snapshots 3 --hardware-drift 0.01 --hardware-crosstalk 0.02 --hardware-snapshot-spacing 75000.0 --hardware-directional --seeds 13 17 23 29 31 37 41 --qasm-root "$QASMBENCH_ROOT" --selection-seed 11 --require-qasmbench && $PYTHON_BIN scripts/run.py invariants && $PYTHON_BIN scripts/run.py validate-proxy-extended --include-weighted --max-circuits 120 --max-qubits 32 --shots 4096 --qasm-root "$QASMBENCH_ROOT" --selection-seed 11 --require-qasmbench && $PYTHON_BIN scripts/run.py verdict
```

If `nvidia-smi` is absent, `print-mega-command` swaps `HARDWARE_DRAWS=1000` for `HARDWARE_DRAWS=200` but keeps the rest identical.

## Legacy experiments
IL/RL/residual work remains under `experiments/legacy_rl/` for reference only. The main story is **Weighted SABRE + gauntlet + verdict**.
