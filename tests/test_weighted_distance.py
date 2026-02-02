import math

import networkx as nx
import pytest

from quantum_routing_rl.hardware.model import EdgeProps, HardwareModel, QubitProps
from quantum_routing_rl.routing.weighted_distance import (
    WeightedDistanceCache,
    WeightedDistanceParams,
)


def _qubit_props() -> dict[int, QubitProps]:
    return {
        q: QubitProps(
            t1_ns=1e6,
            t2_ns=1e6,
            p1_error=1e-4,
            readout_error=1e-3,
            p1_duration_ns=50.0,
        )
        for q in range(3)
    }


def test_weighted_distance_is_deterministic():
    graph = nx.path_graph(3)
    hardware = HardwareModel.synthetic(
        graph, seed=123, directional=True, drift_rate=0.02, snapshots=2
    )
    params = WeightedDistanceParams(alpha_time=0.3, beta_xtalk=0.2)

    cache = WeightedDistanceCache(hardware, params)
    first = cache.dist(0, 2, snapshot_id=1)
    second = cache.dist(0, 2, snapshot_id=1)

    cache_again = WeightedDistanceCache(hardware, params)
    third = cache_again.dist(0, 2, snapshot_id=1)

    assert first == pytest.approx(second)
    assert first == pytest.approx(third)


def test_unit_weights_reduce_to_hop_count():
    p_error = 1.0 - math.exp(-1.0)  # gives base weight ~= 1.0
    edge_props = {
        (0, 1): EdgeProps(p2_error=p_error, t2_duration_ns=1.0),
        (1, 2): EdgeProps(p2_error=p_error, t2_duration_ns=1.0),
    }
    hardware = HardwareModel(
        graph_id="line3",
        adjacency=[(0, 1), (1, 2)],
        edge_props=edge_props,
        qubit_props=_qubit_props(),
        directional_mode=False,
    )
    cache = WeightedDistanceCache(hardware, WeightedDistanceParams(alpha_time=0.0, beta_xtalk=0.0))

    assert cache.dist(0, 1, snapshot_id=0) == pytest.approx(1.0)
    assert cache.dist(0, 2, snapshot_id=0) == pytest.approx(2.0)


def test_symmetry_when_directional_disabled():
    edge_props = {
        (0, 1): EdgeProps(p2_error=0.01, t2_duration_ns=100.0),
        (1, 2): EdgeProps(p2_error=0.02, t2_duration_ns=150.0),
    }
    hardware = HardwareModel(
        graph_id="line3",
        adjacency=[(0, 1), (1, 2)],
        edge_props=edge_props,
        qubit_props=_qubit_props(),
        directional_mode=False,
    )
    cache = WeightedDistanceCache(hardware, WeightedDistanceParams(alpha_time=0.1, beta_xtalk=0.0))

    assert cache.dist(0, 1, snapshot_id=0) == pytest.approx(cache.dist(1, 0, snapshot_id=0))
    assert cache.dist(0, 2, snapshot_id=0) == pytest.approx(cache.dist(2, 0, snapshot_id=0))
