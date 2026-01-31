import networkx as nx

from quantum_routing_rl.hardware.model import HardwareModel


def test_synthetic_is_deterministic():
    graph = nx.path_graph(4)
    hw_a = HardwareModel.synthetic(graph, seed=123)
    hw_b = HardwareModel.synthetic(graph, seed=123)
    assert hw_a.edge_props == hw_b.edge_props
    assert hw_a.qubit_props == hw_b.qubit_props
    assert hw_a.graph_id == hw_b.graph_id


def test_synthetic_ranges_reasonable():
    graph = nx.cycle_graph(5)
    hw = HardwareModel.synthetic(graph, seed=7)

    for props in hw.edge_props.values():
        assert 0.0 < props.p2_error < 0.051
        assert 200.0 <= props.t2_duration_ns <= 600.0

    for props in hw.qubit_props.values():
        assert 20_000.0 <= props.t1_ns <= 200_000.0
        assert 0.0 < props.t2_ns <= props.t1_ns
        assert 1e-4 <= props.p1_error <= 5e-3
        assert 0.01 <= props.readout_error <= 0.05
        assert 20.0 <= props.p1_duration_ns <= 80.0


def test_synthetic_directional_snapshots_are_deterministic():
    graph = nx.path_graph(3)
    hw_a = HardwareModel.synthetic(
        graph, seed=11, directional=True, drift_rate=0.05, snapshots=2, snapshot_spacing_ns=40_000
    )
    hw_b = HardwareModel.synthetic(
        graph, seed=11, directional=True, drift_rate=0.05, snapshots=2, snapshot_spacing_ns=40_000
    )
    assert len(hw_a.snapshots) == 2
    assert len(hw_b.snapshots) == 2
    assert hw_a.snapshots[0].edge_props == hw_b.snapshots[0].edge_props
    assert hw_a.snapshots[1].edge_props == hw_b.snapshots[1].edge_props
    assert hw_a.snapshot_index_for_time(0) == 0
    assert hw_a.snapshot_index_for_time(50_000) == 1
