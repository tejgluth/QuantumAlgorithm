import networkx as nx

from quantum_routing_rl.benchmarks.topologies import (
    coupling_maps_for,
    tiered_topologies,
    topology_registry,
)


def test_grid_5x5_connected_and_size():
    cmap = coupling_maps_for(["grid_5x5"])["grid_5x5"]
    assert int(cmap.size()) >= 25
    g = nx.Graph()
    g.add_edges_from(cmap.get_edges())
    assert nx.is_connected(g)


def test_heavy_hex_27_like_connected():
    cmap = coupling_maps_for(["heavy_hex_27_like"])["heavy_hex_27_like"]
    g = nx.Graph()
    g.add_edges_from(cmap.get_edges())
    assert len(g.nodes()) == 27
    assert nx.is_connected(g)


def test_sparse_32_variants_stable():
    reg = topology_registry()
    undir = reg["sparse_32_undir"]
    dir_cmap = reg["sparse_32_dir"]

    g_undir = nx.Graph()
    g_undir.add_edges_from(undir.get_edges())
    assert g_undir.number_of_nodes() >= 32
    assert nx.is_connected(g_undir)

    g_dir = nx.DiGraph()
    g_dir.add_edges_from(dir_cmap.get_edges())
    assert g_dir.number_of_nodes() >= 32
    assert nx.is_strongly_connected(g_dir)


def test_tiered_topologies_small_subset():
    small = tiered_topologies("small")
    assert {"ring_8", "grid_3x3", "heavy_hex_15"} <= set(small.keys())
