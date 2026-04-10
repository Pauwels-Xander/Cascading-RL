# Data Layout

## Structure

- `processed/` — cleaned edge-list CSVs ready for the evaluation scripts (gitignored; generated locally)

## Real-world datasets

Download with:

    python scripts/download_real_world_data.py

This produces:

| File | Dataset | Nodes | Edges | Source |
|------|---------|-------|-------|--------|
| `processed/ieee300_edges.csv` | IEEE 300-bus power grid | 300 | ~409 | PGLIB-OPF (CC-BY 4.0) |
| `processed/watts_strogatz_edges.csv` | Watts-Strogatz small-world | 300 | 600 | Generated via NetworkX |

## Notes on the cascade model

The cascade model used in evaluation (`load = degree`, `capacity = (1+alpha)*degree`)
is a **stylized approximation** — not a physics-accurate power flow simulation.
Results on real-world topologies test structural generalisation only.

