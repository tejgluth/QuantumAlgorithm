import numpy as np
import pandas as pd

from quantum_routing_rl.eval.paired_deltas import bootstrap_ci, compute_paired_deltas


def test_compute_paired_deltas_basic():
    rows = [
        {
            "baseline_name": "weighted_sabre",
            "circuit_id": "c1",
            "graph_id": "g1",
            "seed": 1,
            "hardware_seed": 10,
            "overall_log_success": -1.0,
        },
        {
            "baseline_name": "qiskit_sabre_best",
            "circuit_id": "c1",
            "graph_id": "g1",
            "seed": 1,
            "hardware_seed": 10,
            "overall_log_success": -1.5,
        },
        {
            "baseline_name": "weighted_sabre",
            "circuit_id": "c2",
            "graph_id": "g1",
            "seed": 2,
            "hardware_seed": 11,
            "overall_log_success": -0.5,
        },
        {
            "baseline_name": "qiskit_sabre_best",
            "circuit_id": "c2",
            "graph_id": "g1",
            "seed": 2,
            "hardware_seed": 11,
            "overall_log_success": -1.0,
        },
    ]
    df = pd.DataFrame(rows)
    deltas = compute_paired_deltas(df)
    assert len(deltas) == 2
    assert np.isclose(deltas["delta_success"].iloc[0], 0.5)
    assert np.isclose(deltas["delta_success"].iloc[1], 0.5)


def test_bootstrap_ci_is_deterministic():
    vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    ci1 = bootstrap_ci(vals, n_resamples=500, seed=42)
    ci2 = bootstrap_ci(vals, n_resamples=500, seed=42)
    assert np.allclose(ci1, ci2)
