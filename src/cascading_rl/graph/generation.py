from __future__ import annotations

from collections.abc import Iterable
from random import Random

import networkx as nx


def make_ba_graph(n: int = 40, m: int = 2, seed: int | None = None) -> nx.Graph:
    """Generate a Barabasi-Albert graph used for synthetic training data."""
    if n < 2:
        raise ValueError("n must be at least 2.")
    if m < 1:
        raise ValueError("m must be at least 1.")
    if m >= n:
        raise ValueError("m must be smaller than n for a BA graph.")
    return nx.barabasi_albert_graph(n=n, m=m, seed=seed)


def make_er_graph(n: int = 40, m: int = 2, seed: int | None = None) -> nx.Graph:
    """Generate an Erdos-Renyi graph with edge probability matched to BA average degree.

    Edge probability p = 2*m / n so that E[degree] = 2*m, matching the BA graph
    used in training. This makes the two graph types comparable in density.
    Retries until a connected graph is produced (ER can be disconnected at low p).
    """
    if n < 2:
        raise ValueError("n must be at least 2.")
    if m < 1:
        raise ValueError("m must be at least 1.")
    p = min(2 * m / n, 1.0)
    rng = Random(seed)
    # Retry until connected; probability of disconnection is low for p = 2m/n >= 0.1
    for attempt in range(1000):
        g = nx.erdos_renyi_graph(n=n, p=p, seed=rng.randint(0, 10**9))
        if nx.is_connected(g):
            return g
    # Fallback: add edges to the largest component until connected
    g = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    components = sorted(nx.connected_components(g), key=len, reverse=True)
    for component in components[1:]:
        node = next(iter(component))
        target = next(iter(components[0]))
        g.add_edge(node, target)
    return g


def make_graph_batch(
    num_graphs: int = 32,
    n_range: tuple[int, int] = (30, 50),
    m: int = 2,
    seed: int | None = None,
    graph_type: str = "ba",
) -> list[nx.Graph]:
    """Generate a batch of synthetic graphs with varying sizes.

    Parameters
    ----------
    graph_type : "ba" (Barabasi-Albert, default) or "er" (Erdos-Renyi).
        ER graphs use edge probability p = 2*m/n to match BA average degree.
    """
    if num_graphs < 1:
        raise ValueError("num_graphs must be at least 1.")
    min_n, max_n = n_range
    if min_n > max_n:
        raise ValueError("n_range must be ordered as (min_n, max_n).")
    if graph_type not in ("ba", "er"):
        raise ValueError(f"graph_type must be 'ba' or 'er', got '{graph_type}'.")

    make_fn = make_ba_graph if graph_type == "ba" else make_er_graph
    rng = Random(seed)
    graphs: list[nx.Graph] = []
    for graph_index in range(num_graphs):
        graph_size = rng.randint(min_n, max_n)
        graph_seed = rng.randint(0, 10**9)
        graph = make_fn(n=graph_size, m=m, seed=graph_seed)
        graph.graph["graph_index"] = graph_index
        graphs.append(graph)
    return graphs


def relabel_graph_with_prefix(graph: nx.Graph, prefix: str) -> nx.Graph:
    """Return a copy with node names prefixed for easier dataset composition."""
    return nx.relabel_nodes(graph, {node: f"{prefix}{node}" for node in graph.nodes()})


def merge_graphs(graphs: Iterable[nx.Graph]) -> nx.Graph:
    """Compose several graphs into one disconnected test graph."""
    merged = nx.Graph()
    for graph in graphs:
        merged = nx.compose(merged, graph)
    return merged
