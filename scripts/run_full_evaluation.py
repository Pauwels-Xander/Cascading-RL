"""Run the complete evaluation suite across all parameter combinations.

Three tiers, all driven by the same (alpha, pfail, budget) grid:

  Tier 1a — In-distribution param grid  (evaluate_param_generalization.py)
             Accepts the full grid as lists and loops internally over all 100
             cells. One script invocation covers all cells.
             Output: experiments/eval_param_generalization/

  Tier 1b — Topology ablation           (evaluate_topology_ablation.py)
             BA vs ER vs WS at n~[20,50]. One invocation per cell.
             Output: experiments/eval_topology_ablation/a{alpha}_p{pfail}_b{budget}/

  Tier 2  — OOD real-world              (evaluate_real_world.py)
             IEEE 300-bus. One invocation per cell.
             Output: experiments/eval_real_world/a{alpha}_p{pfail}_b{budget}/

Grid (matches evaluate_param_generalization.py defaults)
---------------------------------------------------------
  alpha  : [0.10, 0.15, 0.20, 0.25, 0.30]
  pfail  : [0.05, 0.10, 0.15, 0.20, 0.25]
  budget : [1, 2, 3, 4]
  Total  : 100 cells

Usage
-----
    python scripts/run_full_evaluation.py
    python scripts/run_full_evaluation.py --alpha 0.20 0.25 --pfail 0.15 0.20 --budget 1 2
    python scripts/run_full_evaluation.py --skip-indist   # skip Tier 1a
    python scripts/run_full_evaluation.py --skip-topo     # skip Tier 1b
    python scripts/run_full_evaluation.py --skip-ood      # skip Tier 2
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ALPHA  = [0.10, 0.15, 0.20, 0.25, 0.30]
DEFAULT_PFAIL  = [0.05, 0.10, 0.15, 0.20, 0.25]
DEFAULT_BUDGET = [1, 2, 3, 4]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full evaluation suite across parameter grid.")
    p.add_argument("--checkpoint", type=Path,
                   default=ROOT / "experiments" / "learner" / "recovery_q.pt")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "default.yaml")
    p.add_argument("--alpha",  type=float, nargs="+", default=DEFAULT_ALPHA)
    p.add_argument("--pfail",  type=float, nargs="+", default=DEFAULT_PFAIL)
    p.add_argument("--budget", type=int,   nargs="+", default=DEFAULT_BUDGET)
    p.add_argument("--num-graphs", type=int, default=100,
                   help="Graphs per cell for Tiers 1a and 1b (default: 100).")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                   help="Failure seeds for in-dist and topo ablation (default: 0..9).")
    p.add_argument("--ood-seeds", type=int, nargs="+", default=list(range(20)),
                   help="Failure seeds for OOD real-world (default: 0..19).")
    p.add_argument("--skip-indist", action="store_true", help="Skip Tier 1a.")
    p.add_argument("--skip-topo",   action="store_true", help="Skip Tier 1b.")
    p.add_argument("--skip-ood",    action="store_true", help="Skip Tier 2.")
    return p.parse_args()


def _cell_tag(alpha: float, pfail: float, budget: int) -> str:
    return f"a{alpha}_p{pfail}_b{budget}"


def run(cmd: list) -> None:
    """Run a subprocess command, streaming output, exit on failure."""
    print(f"\n>>> {' '.join(str(c) for c in cmd)}\n", flush=True)
    result = subprocess.run([sys.executable] + [str(c) for c in cmd])
    if result.returncode != 0:
        print(f"\nERROR: command exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    args = parse_args()
    grid = list(itertools.product(args.alpha, args.pfail, args.budget))
    total_cells = len(grid)
    seeds_str     = [str(s) for s in args.seeds]
    ood_seeds_str = [str(s) for s in args.ood_seeds]
    alpha_str  = [str(a) for a in args.alpha]
    pfail_str  = [str(p) for p in args.pfail]
    budget_str = [str(b) for b in args.budget]

    print(f"Grid: {len(args.alpha)} alpha x {len(args.pfail)} pfail x "
          f"{len(args.budget)} budget = {total_cells} cells")
    print(f"Checkpoint: {args.checkpoint}")

    # ------------------------------------------------------------------
    # Tier 1a: in-distribution param grid
    # evaluate_param_generalization.py accepts full lists and iterates
    # over all (alpha, pfail, budget) cells internally — one invocation
    # covers all 100 cells.
    # ------------------------------------------------------------------
    if not args.skip_indist:
        print(f"\n{'='*60}")
        print(f"TIER 1a — In-distribution grid ({total_cells} cells, single run)")
        print(f"  alpha  = {args.alpha}")
        print(f"  pfail  = {args.pfail}")
        print(f"  budget = {args.budget}")
        print(f"{'='*60}")
        run([
            "scripts/evaluate_param_generalization.py",
            "--checkpoint", args.checkpoint,
            "--config",     args.config,
            "--alpha",      *alpha_str,
            "--pfail",      *pfail_str,
            "--budget",     *budget_str,
            "--num-graphs", str(args.num_graphs),
            "--seeds",      *seeds_str,
        ])

    # ------------------------------------------------------------------
    # Tier 1b: topology ablation — BA vs ER vs WS
    # Each script call handles one (alpha, pfail, budget) cell.
    # ------------------------------------------------------------------
    if not args.skip_topo:
        print(f"\n{'='*60}")
        print(f"TIER 1b — Topology ablation ({total_cells} cells)")
        print(f"{'='*60}")
        for idx, (alpha, pfail, budget) in enumerate(grid, 1):
            tag = _cell_tag(alpha, pfail, budget)
            out = ROOT / "experiments" / "eval_topology_ablation" / tag
            print(f"\n[{idx}/{total_cells}] {tag}")
            run([
                "scripts/evaluate_topology_ablation.py",
                "--checkpoint", args.checkpoint,
                "--config",     args.config,
                "--alpha",      str(alpha),
                "--pfail",      str(pfail),
                "--budget",     str(budget),
                "--num-graphs", str(args.num_graphs),
                "--seeds",      *seeds_str,
                "--output-dir", out,
            ])

    # ------------------------------------------------------------------
    # Tier 2: OOD real-world (IEEE 300-bus)
    # Each script call handles one (alpha, pfail, budget) cell.
    # ------------------------------------------------------------------
    if not args.skip_ood:
        print(f"\n{'='*60}")
        print(f"TIER 2 — OOD real-world / IEEE 300-bus ({total_cells} cells)")
        print(f"{'='*60}")
        for idx, (alpha, pfail, budget) in enumerate(grid, 1):
            tag = _cell_tag(alpha, pfail, budget)
            out = ROOT / "experiments" / "eval_real_world" / tag
            print(f"\n[{idx}/{total_cells}] {tag}")
            run([
                "scripts/evaluate_real_world.py",
                "--checkpoint", args.checkpoint,
                "--config",     args.config,
                "--alpha",      str(alpha),
                "--pfail",      str(pfail),
                "--budget",     str(budget),
                "--seeds",      *ood_seeds_str,
                "--output-dir", out,
            ])

    # ------------------------------------------------------------------
    # Plots: generate one figure set per (alpha, pfail, budget) cell
    # where both topo ablation and OOD results are available.
    # ------------------------------------------------------------------
    if not args.skip_topo and not args.skip_ood:
        print(f"\n{'='*60}")
        print("PLOTS — topology ablation + OOD per cell")
        print(f"{'='*60}")
        for alpha, pfail, budget in grid:
            tag = _cell_tag(alpha, pfail, budget)
            topo_json = ROOT / "experiments" / "eval_topology_ablation" / tag / "topology_ablation_summary.json"
            ood_json  = ROOT / "experiments" / "eval_real_world" / tag / "ieee300" / "evaluation_summary.json"
            out       = ROOT / "experiments" / "eval_plots" / tag
            if not topo_json.exists() or not ood_json.exists():
                print(f"  SKIP {tag}: missing result files")
                continue
            run([
                "scripts/plot_evaluation_tiers.py",
                "--topo-json", topo_json,
                "--ood-json",  ood_json,
                "--out-dir",   out,
                "--tag",       tag,
            ])

    print(f"\n{'='*60}")
    print("All evaluations complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
