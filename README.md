# quantum-routing-rl

Noise-aware quantum circuit routing with a production-ready **Weighted SABRE** baseline and a reproducible gauntlet harness.

## What’s included
- Weighted SABRE router with hardware-aware distances and multi-trial selection.
- Fair Qiskit baselines (BasicSwap, LookaheadSwap, SABRE best-of-N, preset opt3) plus optional external adapters.
- Benchmark suites (gauntlet small/full/industrial) spanning QASMBench tiers, structured hard synthetics, and richer topologies (heavy-hex, grid_5x5, sparse_32).
- Validation tools: Aer proxy cross-check (`proxy_validation_extended`) and invariant checks (`invariants`).

## Quickstart
1) `make setup`  
2) `make lint` && `make test`  
3) `make gauntlet-small HARDWARE_DRAWS=10` (writes to `artifacts/gauntlet/<timestamp>/`)  
4) `make validate-proxy-extended` and `make invariants` to sanity-check the proxy and outputs.

Gauntlet entrypoints live in `src/quantum_routing_rl/eval/gauntlet.py` and reuse the core harness in `eval/run_eval.py`.

## Minimal API
```python
from qiskit.transpiler import CouplingMap
from quantum_routing_rl import (
    route_with_weighted_sabre,
    WeightedDistanceParams,
    WeightedSabreWeights,
)
from quantum_routing_rl.hardware.model import HardwareModel

cmap = CouplingMap([[0, 1], [1, 2]])
hardware = HardwareModel.synthetic(cmap.graph, seed=7, profile="realistic")
params = WeightedDistanceParams(alpha_time=0.5, beta_xtalk=0.2)
weights = WeightedSabreWeights(lookahead_weight=0.5, decay_weight=0.25, stagnation_weight=0.25, decay_factor=0.9)

result = route_with_weighted_sabre(
    circuit,
    cmap,
    hardware_model=hardware,
    trials=8,
    distance_params=params,
    snapshot_mode="avg",
    router_weights=weights,
)
print(result.metrics.overall_log_success)
```

## Legacy experiments
IL/RL/residual exploration is retained only for reference under `experiments/legacy_rl/` and is **not** part of the gauntlet story or default exports.

## Contributing
- `make lint` / `make test` before PRs.
- Keep artifacts reproducible; never fabricate results or hide fallbacks (see `invariants` target).

## License & citation
Apache-2.0. Cite via `CITATION.cff`.
