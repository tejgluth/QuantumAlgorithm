import pandas as pd

from quantum_routing_rl.eval import proxy_validation_extended as pve


def test_proxy_validation_extended_smoke(tmp_path):
    out = tmp_path / "proxy_validation"
    exit_code = pve.main(
        [
            "--out",
            str(out),
            "--max-circuits",
            "1",
            "--max-qubits",
            "4",
            "--shots",
            "10",
            "--selection-seed",
            "2",
            "--qasm-root",
            "tests/fixtures/qasmbench",
            "--include-weighted",
        ]
    )
    assert exit_code == 0
    skipped = out / "skipped.txt"
    if skipped.exists():
        assert "Aer" in skipped.read_text()
    else:
        corr_path = out / "correlations.csv"
        assert corr_path.exists()
        df = pd.read_csv(corr_path)
        assert not df.empty
