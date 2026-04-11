"""Download and process real-world network datasets into the data/ directory.

Run once before evaluate_real_world.py:

    python scripts/download_real_world_data.py

Datasets
--------
ieee300
    IEEE 300-bus power systems test case from PGLIB-OPF (the official IEEE PES
    benchmark repository). Source: https://github.com/power-grid-lib/pglib-opf
    File: pglib_opf_case300_ieee.m (MATPOWER format, plain text).
    License: Creative Commons Attribution 4.0 (CC-BY 4.0).
    Reference: Babaeinejadsarookolaee et al. (2019), arXiv:1908.02788.

watts_strogatz
    Watts-Strogatz small-world graph (n=300, k=4, p=0.1, seed=42).
    Generated deterministically via NetworkX — no download required.
    Reference: Watts & Strogatz (1998), Nature 393:440-442.

Saved as edge-list CSV files under data/processed/.
"""

from __future__ import annotations

import csv
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PROCESSED_DIR = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# IEEE 300-bus
# ---------------------------------------------------------------------------
IEEE300_URL = (
    "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/"
    "master/pglib_opf_case300_ieee.m"
)
IEEE300_OUT = PROCESSED_DIR / "ieee300_edges.csv"


def _download(url: str) -> str:
    """
    Download the content at the given URL and return it as text.
    
    Returns:
        The response body as decoded text.
    """
    import subprocess
    print(f"  Downloading {url} ...", flush=True)
    result = subprocess.run(
        ["curl", "-fsSL", "--max-time", "30", url],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def parse_ieee300(matpower_text: str) -> list[tuple[int, int]]:
    """
    Extracts undirected branch edges from MATPOWER .m file text and reindexes bus IDs to contiguous 0..N-1 integers.
    
    Parameters:
        matpower_text (str): The full text content of a MATPOWER-format `.m` file containing an `mpc.branch` table.
    
    Returns:
        list[tuple[int, int]]: List of unique edges as `(u, v)` tuples with `u < v`. Each tuple refers to reindexed node IDs in the range 0..N-1. Self-loops are omitted and parallel/duplicate edges are deduplicated, preserving first-seen order.
    """
    raw_edges: list[tuple[int, int]] = []
    in_branch = False
    for line in matpower_text.splitlines():
        stripped = line.strip()
        if re.match(r"mpc\.branch\s*=\s*\[", stripped):
            in_branch = True
            continue
        if in_branch:
            if stripped.startswith("]"):
                break
            if stripped.startswith("%") or not stripped:
                continue
            stripped = stripped.split("%")[0].rstrip("; \t")
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    from_bus = int(float(parts[0]))
                    to_bus = int(float(parts[1]))
                    if from_bus != to_bus:
                        raw_edges.append((from_bus, to_bus))
                except ValueError:
                    continue

    all_buses = sorted({b for edge in raw_edges for b in edge})
    bus_to_idx = {bus: idx for idx, bus in enumerate(all_buses)}
    edges = list(dict.fromkeys(
        (min(bus_to_idx[u], bus_to_idx[v]), max(bus_to_idx[u], bus_to_idx[v]))
        for u, v in raw_edges
    ))
    return edges


def download_ieee300() -> None:
    """
    Download the IEEE 300-bus MATPOWER file, parse its branch connections, and write an undirected edge-list CSV to the processed data directory.
    
    Parses the downloaded MATPOWER text to extract branch endpoint pairs, reindexes bus IDs to contiguous 0..N-1 indices, deduplicates undirected edges, and writes them to IEEE300_OUT with a header ["from", "to"].
    
    Raises:
        RuntimeError: If no edges are parsed from the downloaded MATPOWER file.
    """
    print("IEEE 300-bus power grid:")
    text = _download(IEEE300_URL)
    edges = parse_ieee300(text)
    if not edges:
        raise RuntimeError(
            "No edges parsed from IEEE 300-bus file. "
            "Check that the URL is still valid and the format is MATPOWER."
        )
    num_nodes = max(max(u, v) for u, v in edges) + 1
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with IEEE300_OUT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["from", "to"])
        writer.writerows(edges)
    print(f"  Saved {len(edges)} edges, {num_nodes} nodes -> {IEEE300_OUT}")


# ---------------------------------------------------------------------------
# Watts-Strogatz small-world (generated, no download needed)
# ---------------------------------------------------------------------------
WS_OUT = PROCESSED_DIR / "watts_strogatz_edges.csv"
WS_N = 300
WS_K = 4
WS_P = 0.1
WS_SEED = 42


def generate_watts_strogatz() -> None:
    """
    Generate a deterministic Watts–Strogatz small-world graph and write its undirected edge list to WS_OUT.
    
    Creates a Watts–Strogatz graph using module-level parameters (WS_N, WS_K, WS_P, WS_SEED), ensures the processed data directory exists, and writes a CSV file with header "from,to" containing each undirected edge (endpoints ordered as (min, max)). Prints a brief summary including edge count, node count, average degree, and the output path.
    """
    import networkx as nx
    print(f"Watts-Strogatz small-world (n={WS_N}, k={WS_K}, p={WS_P}, seed={WS_SEED}):")
    g = nx.watts_strogatz_graph(n=WS_N, k=WS_K, p=WS_P, seed=WS_SEED)
    edges = sorted((min(u, v), max(u, v)) for u, v in g.edges())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with WS_OUT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["from", "to"])
        writer.writerows(edges)
    avg_degree = 2 * len(edges) / WS_N
    print(f"  Generated {len(edges)} edges, {WS_N} nodes, avg_degree={avg_degree:.2f} -> {WS_OUT}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrates preparation of real-world network datasets by running the IEEE300 download/parsing and the Watts–Strogatz generation tasks.
    
    Runs the dataset-specific functions in sequence and terminates the process with exit code 1 if any task raises an exception.
    """
    print("Setting up network datasets...\n")
    try:
        download_ieee300()
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print()
    try:
        generate_watts_strogatz()
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\nDone. Run scripts/evaluate_real_world.py to evaluate the trained policy.")


if __name__ == "__main__":
    main()
