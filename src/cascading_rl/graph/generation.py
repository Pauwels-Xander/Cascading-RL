from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
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


def make_ws_graph(n: int = 40, m: int = 2, p: float = 0.1, seed: int | None = None) -> nx.Graph:
    """Generate a Watts-Strogatz small-world graph with average degree matched to BA.

    Uses k = 2*m nearest neighbours (giving average degree 2m = that of BA/ER graphs)
    and rewiring probability p=0.1 to produce small-world structure (high clustering,
    short paths) while preserving connectivity.

    Parameters
    ----------
    n : number of nodes
    m : BA-equivalent parameter; k = 2*m neighbours in the ring lattice
    p : rewiring probability (default 0.1)
    seed : RNG seed for reproducibility
    """
    if n < 2:
        raise ValueError("n must be at least 2.")
    if m < 1:
        raise ValueError("m must be at least 1.")
    k = 2 * m  # ring neighbours, gives average degree = k = 2m
    if k >= n:
        raise ValueError(f"k=2*m={k} must be less than n={n}.")
    return nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)


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
    graph_type : "ba" (Barabasi-Albert, default), "er" (Erdos-Renyi), or "ws" (Watts-Strogatz).
        All types target average degree 2*m:
        - BA : preferential attachment, scale-free degree distribution
        - ER : random, p = 2*m/n
        - WS : small-world ring lattice with k=2*m neighbours, p=0.1 rewiring
    """
    if num_graphs < 1:
        raise ValueError("num_graphs must be at least 1.")
    min_n, max_n = n_range
    if min_n > max_n:
        raise ValueError("n_range must be ordered as (min_n, max_n).")
    if graph_type not in ("ba", "er", "ws"):
        raise ValueError(f"graph_type must be 'ba', 'er', or 'ws', got '{graph_type}'.")

    if graph_type == "ba":
        make_fn = make_ba_graph
    elif graph_type == "er":
        make_fn = make_er_graph
    else:
        make_fn = make_ws_graph
    rng = Random(seed)
    graphs: list[nx.Graph] = []
    for graph_index in range(num_graphs):
        graph_size = rng.randint(min_n, max_n)
        graph_seed = rng.randint(0, 10**9)
        graph = make_fn(n=graph_size, m=m, seed=graph_seed)
        graph.graph["graph_index"] = graph_index
        graphs.append(graph)
    return graphs


def load_real_world_graph(name: str, data_dir: Path | str | None = None) -> nx.Graph:
    """Load a pre-downloaded real-world network from data/processed/.

    Parameters
    ----------
    name : "ieee300" or "usair"
        Which dataset to load.
    data_dir : path to the data/processed/ directory. Defaults to the repo's
        data/processed/ folder resolved relative to this file.

    Returns
    -------
    A connected, undirected NetworkX graph with 0-indexed integer nodes.
    Raises FileNotFoundError if the CSV has not been downloaded yet —
    run scripts/download_real_world_data.py first.
    """
    filenames = {
        "ieee300": "ieee300_edges.csv",
        "watts_strogatz": "watts_strogatz_edges.csv",
    }
    if name not in filenames:
        raise ValueError(f"Unknown real-world graph '{name}'. Choose from: {list(filenames)}")

    if data_dir is None:
        # Resolve relative to this file: src/cascading_rl/graph/ -> repo root -> data/processed/
        data_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
    csv_path = Path(data_dir) / filenames[name]

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Real-world graph file not found: {csv_path}\n"
            "Run:  python scripts/download_real_world_data.py"
        )

    edges: list[tuple[int, int]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u, v = int(row["from"]), int(row["to"])
            if u != v:
                edges.append((u, v))

    g = nx.Graph()
    g.add_edges_from(edges)

    if g.number_of_nodes() == 0:
        raise ValueError(
            f"Real-world graph '{name}' loaded from {csv_path} contains no valid edges. "
            "The CSV may be empty or contain only self-loops."
        )

    # Ensure connectivity — keep largest component and re-index 0..N-1
    if not nx.is_connected(g):
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc).copy()
        mapping = {old: new for new, old in enumerate(sorted(g.nodes()))}
        g = nx.relabel_nodes(g, mapping)

    g.graph["name"] = name
    return g


def relabel_graph_with_prefix(graph: nx.Graph, prefix: str) -> nx.Graph:
    """Return a copy with node names prefixed for easier dataset composition."""
    return nx.relabel_nodes(graph, {node: f"{prefix}{node}" for node in graph.nodes()})


def merge_graphs(graphs: Iterable[nx.Graph]) -> nx.Graph:
    """Compose several graphs into one disconnected test graph."""
    merged = nx.Graph()
    for graph in graphs:
        merged = nx.compose(merged, graph)
    return merged
