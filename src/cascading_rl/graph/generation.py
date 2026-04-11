from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from random import Random

import networkx as nx


def make_ba_graph(n: int = 40, m: int = 2, seed: int | None = None) -> nx.Graph:
    """
    Generate a Barabási–Albert (preferential attachment) undirected graph for synthetic data.
    
    Parameters:
        n (int): Number of nodes in the graph; must be at least 2.
        m (int): Number of edges to attach from a new node to existing nodes (attachment parameter); must be at least 1 and smaller than `n`.
        seed (int | None): Optional random seed for reproducible graph generation.
    
    Returns:
        graph (nx.Graph): An undirected Barabási–Albert graph with `n` nodes and attachment parameter `m`.
    
    Raises:
        ValueError: If `n < 2`, if `m < 1`, or if `m >= n`.
    """
    if n < 2:
        raise ValueError("n must be at least 2.")
    if m < 1:
        raise ValueError("m must be at least 1.")
    if m >= n:
        raise ValueError("m must be smaller than n for a BA graph.")
    return nx.barabasi_albert_graph(n=n, m=m, seed=seed)


def make_ws_graph(n: int = 40, m: int = 2, p: float = 0.1, seed: int | None = None) -> nx.Graph:
    """
    Generate a Watts–Strogatz small-world graph parameterized to target an average degree of 2*m.
    
    Parameters:
        n (int): Number of nodes; must be at least 2.
        m (int): Controls the ring lattice neighbourhoods; each node is initially connected to k = 2*m nearest neighbours. Must be at least 1 and satisfy 2*m < n.
        p (float): Rewiring probability for each edge (controls randomness / small-world property).
        seed (int | None): Optional random seed for reproducibility.
    
    Returns:
        nx.Graph: An undirected Watts–Strogatz graph with n nodes, initial degree k = 2*m, and rewiring probability p.
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
    """
    Generate an Erdős–Rényi undirected graph whose expected average degree is approximately 2*m and ensure the returned graph is connected.
    
    The edge probability used is p = min(2*m / n, 1.0). The function will attempt up to 1000 independent draws to obtain a connected graph; if no connected graph is produced, it falls back to a single draw and connects remaining components by adding edges between a representative node of each component and the largest component.
    
    Parameters:
        n (int): Number of nodes; must be at least 2.
        m (int): Target parameter such that expected average degree ≈ 2*m; must be at least 1.
        seed (int | None): Optional random seed for reproducible graph generation.
    
    Returns:
        nx.Graph: A connected NetworkX undirected graph with n nodes.
    
    Raises:
        ValueError: If `n < 2` or `m < 1`.
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
    """
    Create a list of synthetic undirected graphs with varying node counts.
    
    Parameters:
        num_graphs (int): Number of graphs to generate; must be >= 1.
        n_range (tuple[int, int]): Inclusive range (min_n, max_n) for the number of nodes per graph; min_n must be <= max_n.
        m (int): Parameter controlling average degree; generators target average degree approximately 2*m.
        seed (int | None): Seed for reproducible variation across the batch.
        graph_type (str): Generator type to use: "ba" (Barabási–Albert), "er" (Erdős–Rényi), or "ws" (Watts–Strogatz).
    
    Returns:
        list[nx.Graph]: Generated NetworkX graphs. Each graph has metadata key "graph_index" set to its position in the returned list.
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
    """
    Load a named real-world undirected graph from a pre-downloaded CSV and return a connected, 0-indexed NetworkX graph.
    
    Parameters:
        name (str): Dataset identifier. Supported values: "ieee300", "watts_strogatz".
        data_dir (Path | str | None): Path to the directory containing processed CSV files. If None, defaults to the repository's data/processed directory resolved relative to this file.
    
    Returns:
        nx.Graph: A connected undirected graph whose nodes are integers reindexed to 0..N-1 and with graph metadata key "name" set to `name`.
    
    Raises:
        ValueError: If `name` is not one of the supported dataset identifiers or if the loaded CSV contains no valid edges.
        FileNotFoundError: If the expected CSV file is not present at the resolved `data_dir`.
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
    """
    Create a copy of the graph with each node label prefixed by the given string.
    
    Parameters:
        prefix (str): String to prepend to each node label; the original label is converted to a string before concatenation.
    
    Returns:
        nx.Graph: A new graph with the same nodes and edges but with node labels replaced by `prefix + original_label`.
    """
    return nx.relabel_nodes(graph, {node: f"{prefix}{node}" for node in graph.nodes()})


def merge_graphs(graphs: Iterable[nx.Graph]) -> nx.Graph:
    """Compose several graphs into one disconnected test graph."""
    merged = nx.Graph()
    for graph in graphs:
        merged = nx.compose(merged, graph)
    return merged
