import pandas as pd
import pytest

from quantum_routing_rl.eval.scoring import PRESSURE_GRAPHS, compute_scalar_objective


def _build_summary(bad_duration: bool = False) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    policy_primary = [-1.0, -1.1, -1.2]
    for idx, graph in enumerate(PRESSURE_GRAPHS):
        duration_policy = 120.0 if not (bad_duration and idx == 0) else 220.0
        rows.append(
            {
                "graph_id": graph,
                "baseline_name": "policy",
                "swaps_inserted_mean": 12.0,
                "twoq_depth_mean": 15.0,
                "total_duration_ns_mean": duration_policy,
                "overall_log_success_mean": policy_primary[idx],
            }
        )
        rows.append(
            {
                "graph_id": graph,
                "baseline_name": "sabre_layout_swap",
                "swaps_inserted_mean": 10.0,
                "twoq_depth_mean": 14.0,
                "total_duration_ns_mean": 110.0,
                "overall_log_success_mean": -1.5,
            }
        )
    return pd.DataFrame(rows)


def test_compute_scalar_objective_passes_constraints():
    df = _build_summary(bad_duration=False)
    result = compute_scalar_objective(df, policy_name="policy")
    assert result.constraints_ok is True
    assert result.missing_policy_graphs == []
    assert result.missing_sabre_graphs == []
    assert result.primary == pytest.approx((-1.0 - 1.1 - 1.2) / 3)
    assert result.mean_ratios["swaps"] == pytest.approx(1.2)


def test_compute_scalar_objective_flags_constraint_violation():
    df = _build_summary(bad_duration=True)
    result = compute_scalar_objective(df, policy_name="policy")
    assert result.constraints_ok is False
    assert result.constraint_details["total_duration_ns"][PRESSURE_GRAPHS[0]] > 1.5
