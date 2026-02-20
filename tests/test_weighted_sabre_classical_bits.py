from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.eval.metrics import assert_coupling_compatible
from quantum_routing_rl.hardware.model import HardwareModel
from quantum_routing_rl.models.weighted_sabre import route_with_weighted_sabre


def test_weighted_sabre_preserves_classical_bits():
    circuit = QuantumCircuit(3, 3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure(0, 0)
    circuit.barrier()
    circuit.cx(1, 2)
    circuit.reset(2)
    circuit.measure(1, 1)
    circuit.measure(2, 2)

    cmap = CouplingMap([[0, 1], [1, 2], [2, 0]])
    hardware = HardwareModel.synthetic(
        cmap, seed=42, directional=True, drift_rate=0.02, snapshots=2
    )

    result = route_with_weighted_sabre(
        circuit,
        cmap,
        hardware_model=hardware,
        seed=3,
        trials=1,
        snapshot_mode="avg",
    )

    assert result.circuit is not None
    assert len(result.circuit.clbits) == circuit.num_clbits
    assert [creg.size for creg in result.circuit.cregs] == [creg.size for creg in circuit.cregs]

    measurements = [inst for inst in result.circuit.data if inst.operation.name == "measure"]
    assert len(measurements) == 3
    for inst in measurements:
        for clbit in inst.clbits:
            assert clbit in result.circuit.clbits

    assert_coupling_compatible(result.circuit, cmap.get_edges())


def test_weighted_sabre_falls_back_on_classical_control():
    circuit = QuantumCircuit.from_qasm_str(
        "\n".join(
            [
                "OPENQASM 2.0;",
                'include "qelib1.inc";',
                "qreg q[2];",
                "creg c[2];",
                "h q[0];",
                "cx q[0],q[1];",
                "measure q[0] -> c[0];",
                "reset q[0];",
                "if(c==1) u1(0.1) q[0];",
            ]
        )
    )

    cmap = CouplingMap([[0, 1], [1, 0]])
    hardware = HardwareModel.synthetic(cmap, seed=7, directional=True, drift_rate=0.01, snapshots=2)

    result = route_with_weighted_sabre(
        circuit,
        cmap,
        hardware_model=hardware,
        seed=3,
        trials=2,
        snapshot_mode="avg",
    )

    assert result.circuit is not None
    assert result.baseline_status == "fallback"
    assert result.fallback_used is True
    assert result.fallback_reason == "UNSUPPORTED_CLASSICAL_CONTROL"
    assert result.extra.get("fallback_source") == "qiskit_sabre_best"
