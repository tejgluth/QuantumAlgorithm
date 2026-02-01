import pandas as pd
import pytest

from quantum_routing_rl.eval.run_eval import _write_summary
from quantum_routing_rl.eval import regression_checks as rc


def test_write_summary_splits_mean_and_std(tmp_path):
    data = [
        {
            "graph_id": "ring_8",
            "baseline_name": "il_soft",
            "swaps_inserted": 10,
            "twoq_count": 20,
            "twoq_depth": 30,
            "depth": 40,
            "routing_runtime_s": 0.5,
            "noise_proxy_score": 0.9,
            "log_success_proxy": -0.1,
            "duration_proxy": 0.2,
            "overall_log_success": -1.0,
            "total_duration_ns": 1000.0,
            "decoherence_penalty": 0.01,
        },
        {
            "graph_id": "ring_8",
            "baseline_name": "il_soft",
            "swaps_inserted": 14,
            "twoq_count": 22,
            "twoq_depth": 32,
            "depth": 42,
            "routing_runtime_s": 0.6,
            "noise_proxy_score": 0.8,
            "log_success_proxy": -0.2,
            "duration_proxy": 0.25,
            "overall_log_success": -0.8,
            "total_duration_ns": 1100.0,
            "decoherence_penalty": 0.02,
        },
    ]
    df = pd.DataFrame(data)
    mean_path = tmp_path / "summary.csv"
    std_path = tmp_path / "summary_std.csv"

    _write_summary(df, mean_path, std_path)

    mean_df = pd.read_csv(mean_path)
    std_df = pd.read_csv(std_path)

    assert set(["graph_id", "baseline_name", "swaps_inserted_mean"]).issubset(mean_df.columns)
    assert set(["graph_id", "baseline_name", "swaps_inserted_std"]).issubset(std_df.columns)
    assert mean_df.loc[0, "swaps_inserted_mean"] == pytest.approx(12.0)
    assert std_df.loc[0, "swaps_inserted_std"] == pytest.approx(2.82842712, rel=1e-6)


def test_ratio_failures_checks_multiple_metrics():
    df = pd.DataFrame(
        [
            {
                "graph_id": "g1",
                "baseline_name": "teacher",
                "swaps_inserted_mean": 10.0,
                "twoq_depth_mean": 20.0,
                "total_duration_ns_mean": 100.0,
            },
            {
                "graph_id": "g1",
                "baseline_name": "il",
                "swaps_inserted_mean": 12.0,
                "twoq_depth_mean": 24.0,
                "total_duration_ns_mean": 120.0,
            },
            {
                "graph_id": "g2",
                "baseline_name": "teacher",
                "swaps_inserted_mean": 10.0,
                "twoq_depth_mean": 20.0,
                "total_duration_ns_mean": 100.0,
            },
            {
                "graph_id": "g2",
                "baseline_name": "il",
                "swaps_inserted_mean": 14.5,
                "twoq_depth_mean": 25.0,
                "total_duration_ns_mean": 145.0,
            },
        ]
    )

    failures = rc._ratio_failures(
        df,
        baseline="il",
        teacher="teacher",
        graphs=["g1", "g2"],
        max_ratio=1.3,
        require_baseline=True,
        metrics=["swaps_inserted", "twoq_depth", "total_duration_ns"],
    )

    assert failures  # g2 swaps + duration should fail
    assert any("g2" in msg and "swaps_inserted" in msg for msg in failures)
    assert any("g2" in msg and "total_duration_ns" in msg for msg in failures)
    assert not any("g1" in msg for msg in failures)
