import numpy as np
import pandas as pd

from quantum_routing_rl.eval.variance import variance_breakdown


def test_variance_breakdown_two_factors():
    rows = []
    # Hardware H1: seeds 1,2
    rows.extend(
        [
            {
                "baseline_name": "weighted_sabre",
                "graph_id": "g1",
                "hardware_seed": 1,
                "seed": 1,
                "overall_log_success": val,
            }
            for val in (1.0, 1.0)
        ]
    )
    rows.extend(
        [
            {
                "baseline_name": "weighted_sabre",
                "graph_id": "g1",
                "hardware_seed": 1,
                "seed": 2,
                "overall_log_success": val,
            }
            for val in (2.0, 2.0)
        ]
    )
    # Hardware H2: seeds 1,2
    rows.extend(
        [
            {
                "baseline_name": "weighted_sabre",
                "graph_id": "g1",
                "hardware_seed": 2,
                "seed": 1,
                "overall_log_success": val,
            }
            for val in (3.0, 3.0)
        ]
    )
    rows.extend(
        [
            {
                "baseline_name": "weighted_sabre",
                "graph_id": "g1",
                "hardware_seed": 2,
                "seed": 2,
                "overall_log_success": val,
            }
            for val in (4.0, 4.0)
        ]
    )

    df = pd.DataFrame(rows)
    breakdown = variance_breakdown(df, center_by_circuit=False)
    assert len(breakdown) == 1
    row = breakdown.iloc[0]
    # Expected sums of squares: total=10, between_hw=8, between_seed=2, residual=0
    denom = len(df) - 1
    assert np.isclose(row["total_variance"], 10 / denom)
    assert np.isclose(row["between_hardware_variance"], 8 / denom)
    assert np.isclose(row["between_seed_variance"], 2 / denom)
    assert np.isclose(row["residual_variance"], 0.0)
    # Fractions should sum to ~1
    frac_sum = row["between_hardware_frac"] + row["between_seed_frac"] + row["residual_frac"]
    assert np.isclose(frac_sum, 1.0)
