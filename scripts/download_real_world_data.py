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

usair
    US domestic airport route network derived from OpenFlights.org open data.
    Airports: https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat
    Routes:   https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat
    License: OpenFlights data is provided under the Open Database Licence (ODbL).

Both datasets are saved as simple edge-list CSV files under data/processed/.
"""

from __future__ import annotations

import csv
import io
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
    import subprocess
    print(f"  Downloading {url} ...", flush=True)
    # Use system curl — avoids macOS Python SSL certificate issues entirely.
    result = subprocess.run(
        ["curl", "-fsSL", "--max-time", "30", url],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def parse_ieee300(matpower_text: str) -> list[tuple[int, int]]:
    """Extract branch connections from a MATPOWER .m file.

    MATPOWER branch data: each row is [from_bus, to_bus, r, x, b, ...]
    Bus numbers are arbitrary integers in the file; we re-index to 0..N-1.
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

    # Re-index bus numbers to 0..N-1
    all_buses = sorted({b for edge in raw_edges for b in edge})
    bus_to_idx = {bus: idx for idx, bus in enumerate(all_buses)}
    edges = list(dict.fromkeys(
        (min(bus_to_idx[u], bus_to_idx[v]), max(bus_to_idx[u], bus_to_idx[v]))
        for u, v in raw_edges
    ))
    return edges


def download_ieee300() -> None:
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
# US Air Traffic (OpenFlights)
# ---------------------------------------------------------------------------
AIRPORTS_URL = (
    "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
)
ROUTES_URL = (
    "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"
)
USAIR_OUT = PROCESSED_DIR / "usair_edges.csv"

# FAA country code for US airports
_US_COUNTRY = "United States"


def _parse_airports(text: str) -> dict[str, int]:
    """Return {IATA_code: node_index} for US airports with valid IATA codes."""
    us_airports: list[str] = []
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        if len(row) < 8:
            continue
        country = row[3].strip().strip('"')
        iata = row[4].strip().strip('"')
        if country == _US_COUNTRY and len(iata) == 3 and iata != "\\N":
            us_airports.append(iata)
    # Stable sort → deterministic node indices
    us_airports = sorted(set(us_airports))
    return {code: idx for idx, code in enumerate(us_airports)}


def _parse_routes(text: str, airport_index: dict[str, int]) -> list[tuple[int, int]]:
    """Return undirected edges between US airports."""
    edges: set[tuple[int, int]] = set()
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        if len(row) < 5:
            continue
        src_iata = row[2].strip().strip('"')
        dst_iata = row[4].strip().strip('"')
        if src_iata in airport_index and dst_iata in airport_index:
            u = airport_index[src_iata]
            v = airport_index[dst_iata]
            if u != v:
                edges.add((min(u, v), max(u, v)))
    return sorted(edges)


def download_usair() -> None:
    import networkx as nx

    print("US Air Traffic (OpenFlights):")
    airports_text = _download(AIRPORTS_URL)
    routes_text = _download(ROUTES_URL)
    airport_index = _parse_airports(airports_text)
    edges = _parse_routes(routes_text, airport_index)
    if not edges:
        raise RuntimeError(
            "No US air routes parsed. Check that the OpenFlights URLs are still valid."
        )

    # Keep only the largest connected component
    g = nx.Graph()
    g.add_nodes_from(range(len(airport_index)))
    g.add_edges_from(edges)
    largest_cc = max(nx.connected_components(g), key=len)
    sub = g.subgraph(largest_cc).copy()
    # Re-index nodes 0..N-1
    mapping = {old: new for new, old in enumerate(sorted(largest_cc))}
    edges_out = sorted(
        (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
        for u, v in sub.edges()
    )
    num_nodes = len(largest_cc)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with USAIR_OUT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["from", "to"])
        writer.writerows(edges_out)
    print(f"  Saved {len(edges_out)} edges, {num_nodes} nodes -> {USAIR_OUT}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Downloading real-world network datasets...\n")
    try:
        download_ieee300()
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)

    print()
    try:
        download_usair()
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)

    print("\nDone. Run scripts/evaluate_real_world.py to evaluate the trained policy.")


if __name__ == "__main__":
    main()
