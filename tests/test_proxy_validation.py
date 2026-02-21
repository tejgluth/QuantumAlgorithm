import pandas as pd
import pytest

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


def test_create_aer_simulator_gpu_fallback(monkeypatch):
    class FakeAer:
        calls = []

        def __init__(self, **kwargs):
            FakeAer.calls.append(kwargs)
            if kwargs.get("device") == "GPU":
                raise RuntimeError("gpu unavailable")

    monkeypatch.setattr(pve, "AerSimulator", FakeAer)
    sim, device = pve._create_aer_simulator(
        noise_model=None,
        device_pref="gpu",
        method="automatic",
        precision="single",
        strict_device=False,
    )
    assert sim is not None
    assert device == "cpu"
    assert len(FakeAer.calls) >= 2
    assert FakeAer.calls[0].get("device") == "GPU"
    assert FakeAer.calls[-1].get("device") == "CPU"


def test_create_aer_simulator_gpu_strict(monkeypatch):
    class FakeAer:
        def __init__(self, **kwargs):
            if kwargs.get("device") == "GPU":
                raise RuntimeError("gpu unavailable")

    monkeypatch.setattr(pve, "AerSimulator", FakeAer)
    with pytest.raises(RuntimeError):
        pve._create_aer_simulator(
            noise_model=None,
            device_pref="gpu",
            method="automatic",
            precision="single",
            strict_device=True,
        )


def test_create_aer_simulator_parallel_options(monkeypatch):
    calls = []

    class FakeAer:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(pve, "AerSimulator", FakeAer)
    sim, device = pve._create_aer_simulator(
        noise_model=None,
        device_pref="cpu",
        method="automatic",
        precision="single",
        strict_device=False,
        max_parallel_threads=8,
        max_parallel_experiments=4,
    )
    assert sim is not None
    assert device == "cpu"
    assert calls
    assert calls[-1].get("max_parallel_threads") == 8
    assert calls[-1].get("max_parallel_experiments") == 4
