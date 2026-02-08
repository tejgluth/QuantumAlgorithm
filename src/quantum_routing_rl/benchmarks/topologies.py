"""Standardized coupling map definitions for gauntlet benchmarks."""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Dict, Iterable

import networkx as nx
from qiskit.transpiler import CouplingMap


def _as_bidirectional(edges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return bidirectional edges for an undirected topology."""
    expanded: list[tuple[int, int]] = []
    for u, v in edges:
        expanded.append((u, v))
        expanded.append((v, u))
    return sorted(set(expanded))


def _grid_3x3() -> CouplingMap:
    edges = [
        (0, 1),
        (1, 2),
        (3, 4),
        (4, 5),
        (6, 7),
        (7, 8),
        (0, 3),
        (3, 6),
        (1, 4),
        (4, 7),
        (2, 5),
        (5, 8),
    ]
    return CouplingMap(_as_bidirectional(edges))


def _grid_5x5() -> CouplingMap:
    """5x5 grid, bidirectional when supported by Qiskit helper."""
    try:
        # Qiskit >=0.46 uses bidirectional kwarg, older versions ignore it safely.
        return CouplingMap.from_grid(5, 5, bidirectional=True)  # type: ignore[arg-type]
    except TypeError:
        return CouplingMap.from_grid(5, 5)
    except Exception:
        # Fallback: build manually to avoid import/runtime failures.
        edges: list[tuple[int, int]] = []
        for r in range(5):
            for c in range(5):
                idx = r * 5 + c
                if c < 4:
                    edges.append((idx, idx + 1))
                if r < 4:
                    edges.append((idx, idx + 5))
        return CouplingMap(_as_bidirectional(edges))


def _ring_8() -> CouplingMap:
    return CouplingMap(_as_bidirectional([(i, (i + 1) % 8) for i in range(8)]))


def _line_3() -> CouplingMap:
    return CouplingMap(_as_bidirectional([(0, 1), (1, 2)]))


def _square_4() -> CouplingMap:
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    return CouplingMap(_as_bidirectional(edges))


def _heavy_hex_15() -> CouplingMap:
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (2, 6),
        (4, 7),
        (5, 6),
        (6, 7),
        (5, 8),
        (6, 9),
        (7, 10),
        (8, 9),
        (9, 10),
        (8, 11),
        (9, 12),
        (10, 13),
        (11, 12),
        (12, 13),
        (11, 14),
        (13, 14),
    ]
    return CouplingMap(_as_bidirectional(edges))


def _heavy_hex_large(distance: int | None = None) -> CouplingMap:
    """Large heavy-hex lattice via Qiskit helper with a safe fallback."""
    candidates = []
    if distance is not None:
        candidates.append(distance)
    candidates.extend([5, 4, 3, 2])
    seen: set[int] = set()
    for dist in candidates:
        if dist in seen:
            continue
        seen.add(dist)
        try:
            return CouplingMap.from_heavy_hex(distance=dist)
        except Exception:
            continue
    # Last resort: reuse heavy_hex_15 so callers still proceed.
    return _heavy_hex_15()


def _heavy_hex_27_like() -> CouplingMap:
    """Connected 27-qubit induced subgraph carved from a larger heavy-hex lattice."""
    base = _heavy_hex_large(distance=5)
    edges = base.get_edges()
    graph = nx.Graph()
    graph.add_edges_from(edges)
    # BFS walk to pick a connected 27-node subset deterministically.
    bfs_nodes = list(nx.bfs_tree(graph, source=0).nodes())[:27]
    sub_edges = [(u, v) for u, v in edges if u in bfs_nodes and v in bfs_nodes]
    sub_graph = nx.Graph()
    sub_graph.add_edges_from(sub_edges)
    if not nx.is_connected(sub_graph):
        # Ensure connectivity by adding spanning-tree edges among selected nodes.
        for u, v in nx.bfs_edges(graph, source=0):
            if u in bfs_nodes and v in bfs_nodes and (u, v) not in sub_edges:
                sub_edges.append((u, v))
            sub_graph = nx.Graph()
            sub_graph.add_edges_from(sub_edges)
            if nx.is_connected(sub_graph):
                break
    return CouplingMap(_as_bidirectional(sub_edges))


def _sparse_graph_32(seed: int = 2024) -> nx.Graph:
    rng = random.Random(seed)
    g = nx.Graph()
    g.add_nodes_from(range(32))
    order = list(range(32))
    rng.shuffle(order)
    # Random spanning tree for baseline connectivity.
    for idx in range(1, len(order)):
        u = order[idx]
        v = order[rng.randrange(idx)]
        g.add_edge(u, v)
    target_edges = 40  # keep sparse but add a little redundancy.
    while g.number_of_edges() < target_edges:
        u, v = rng.sample(range(32), 2)
        g.add_edge(u, v)
    return g


def _sparse_32_undirected(seed: int = 2024) -> CouplingMap:
    g = _sparse_graph_32(seed)
    return CouplingMap(_as_bidirectional(g.edges()))


def _sparse_32_directed(seed: int = 2024) -> CouplingMap:
    base = _sparse_graph_32(seed)
    rng = random.Random(seed + 17)
    oriented: list[tuple[int, int]] = []
    for u, v in sorted(base.edges()):
        oriented.append((u, v) if rng.random() < 0.5 else (v, u))
    dg = nx.DiGraph()
    dg.add_nodes_from(base.nodes())
    dg.add_edges_from(oriented)
    if not nx.is_strongly_connected(dg):
        # Enforce strong connectivity by orienting spanning tree edges both ways.
        for u, v in nx.bfs_edges(base, source=0):
            if not dg.has_edge(u, v):
                oriented.append((u, v))
            if not dg.has_edge(v, u):
                oriented.append((v, u))
        dg = nx.DiGraph()
        dg.add_nodes_from(base.nodes())
        dg.add_edges_from(oriented)
    return CouplingMap(oriented)


@lru_cache(maxsize=1)
def topology_registry() -> Dict[str, CouplingMap]:
    """Return canonical coupling maps keyed by topology id."""
    return {
        "line_3": _line_3(),
        "square_4": _square_4(),
        "ring_8": _ring_8(),
        "grid_3x3": _grid_3x3(),
        "grid_5x5": _grid_5x5(),
        "heavy_hex_15": _heavy_hex_15(),
        "heavy_hex_large": _heavy_hex_large(distance=5),
        "heavy_hex_27_like": _heavy_hex_27_like(),
        "sparse_32_undir": _sparse_32_undirected(),
        "sparse_32_dir": _sparse_32_directed(),
    }


def coupling_maps_for(names: Iterable[str]) -> dict[str, CouplingMap]:
    """Select a subset of registered topologies, raising on unknown ids."""
    reg = topology_registry()
    selected: dict[str, CouplingMap] = {}
    for name in names:
        if name not in reg:
            msg = f"Unknown topology id '{name}'"
            raise KeyError(msg)
        selected[name] = reg[name]
    return selected


def tiered_topologies(tier: str) -> dict[str, CouplingMap]:
    """Convenience chooser for gauntlet tiers."""
    tier = tier.lower()
    if tier == "small":
        return coupling_maps_for(["ring_8", "grid_3x3", "heavy_hex_15"])
    if tier == "medium":
        return coupling_maps_for(["grid_5x5", "heavy_hex_27_like", "sparse_32_undir"])
    if tier in {"large", "industrial"}:
        return coupling_maps_for(["heavy_hex_large", "sparse_32_dir"])
    # Default to all when unknown tier requested.
    return topology_registry()
