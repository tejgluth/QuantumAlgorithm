# Paper Assets

- `paper.md` – publication-style writeup (Markdown).
- `references.bib` – minimal bibliography (SABRE, noise-aware mapping, Qiskit docs).
- `figures/` – symlinks to `artifacts/plots_final/Fig1–Fig5` (png + pdf).
- `tables/` – copies of `artifacts/tables/` including `significance_effect.*`.

## Reproduce the Paper
1. Run `make lint` and `make test`.
2. Generate the main evaluation with the weighted SABRE trials and fair Qiskit trials baseline:
   `SKIP_RL=1 make eval-noise-unguarded-weighted HARDWARE_DRAWS=50`
3. Build tables and figures: `make reproduce-paper`
   - This invokes `quantum_routing_rl.eval.paper_assets` to refresh `artifacts/plots_final/`
     and `artifacts/tables/` and updates `paper/figures` and `paper/tables`.

All commands are reproducible from the repo root; no external data downloads are required beyond bundled QASMBench fixtures.
