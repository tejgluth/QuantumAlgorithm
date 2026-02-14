import warnings

from quantum_routing_rl.benchmarks.qasmbench_loader import load_qasmbench_tier


def _write(path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_qasmbench_loader_suppresses_warnings(tmp_path, capsys):
    valid = tmp_path / "valid.qasm"
    _write(
        valid,
        "\n".join(
            [
                "OPENQASM 2.0;",
                'include "qelib1.inc";',
                "qreg q[1];",
                "creg c[1];",
                "h q[0];",
                "measure q[0] -> c[0];",
            ]
        ),
    )

    # Three invalid files (syntax errors).
    for idx in range(3):
        _write(tmp_path / f"invalid_{idx}.qasm", "OPENQASM 2.0;\nqreg q[1]\ncreg c[1];\n")

    # Three unsupported OpenQASM 3 files.
    for idx in range(3):
        _write(
            tmp_path / f"open3_{idx}.qasm",
            "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\ndefcal X $0 {}\n",
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        circuits = load_qasmbench_tier(tmp_path, tier="small", selection_seed=0)

    assert len(circuits) == 1
    assert circuits[0].path == valid.resolve()

    # Only the first five warnings should be emitted.
    assert len(caught) == 5
    messages = [str(w.message) for w in caught]
    assert any("unsupported dialect" in msg for msg in messages)
    assert any("invalid QASM" in msg for msg in messages)

    stdout = capsys.readouterr().out
    assert "suppressed 1" in stdout
    assert "loaded=1" in stdout
    assert "skipped_invalid=3" in stdout
    assert "skipped_unsupported=3" in stdout
