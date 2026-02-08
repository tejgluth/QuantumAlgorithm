import json
import pandas as pd

from quantum_routing_rl.eval import invariants


def test_invariants_writes_report(tmp_path):
    results = tmp_path / "results.csv"
    data = [
        {
            "circuit_id": "c0",
            "graph_id": "g0",
            "baseline_name": "b0",
            "seed": 1,
            "hardware_seed": 1,
            "swaps_inserted": 2,
            "twoq_count": 3,
            "twoq_depth": 2,
            "depth": 6,
            "overall_log_success": -1.0,
            "total_duration_ns": 10.0,
            "fallback_used": False,
            "baseline_status": "ok",
            "fallback_reason": None,
        },
        {
            "circuit_id": "c0",
            "graph_id": "g0",
            "baseline_name": "b0",
            "seed": 1,
            "hardware_seed": 1,
            "swaps_inserted": 2,
            "twoq_count": 3,
            "twoq_depth": 2,
            "depth": 6,
            "overall_log_success": -1.0,
            "total_duration_ns": 10.0,
            "fallback_used": False,
            "baseline_status": "ok",
            "fallback_reason": None,
        },
    ]
    pd.DataFrame(data).to_csv(results, index=False)

    out_dir = tmp_path / "inv"
    exit_code = invariants.main(["--results", str(results), "--out", str(out_dir)])

    assert exit_code == 0
    assert (out_dir / "report.md").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["passed"] is True
