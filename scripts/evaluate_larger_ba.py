"""Evaluate trained policy on larger BA graphs: n=100, 200, 500, 1000.

Tests scale generalisation beyond the training range (n ∈ [30, 50]).
Budget is scaled proportionally to graph size (same reference_n=40 as training).
Each graph size uses 20 graphs × 10 failure seeds = 200 episodes.

Greedy is excluded: its O(|failed| × steps) per-step rollout cost is
computationally infeasible at n >= 100 (hours per size). This is itself
a meaningful finding reported in the paper.

Output
------
experiments/eval_larger_ba/larger_ba_summary.json
experiments/eval_larger_ba/run_metadata.json

Usage
-----
    python scripts/evaluate_larger_ba.py
    python scripts/evaluate_larger_ba.py --sizes 100 200 500 1000 --num-graphs 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.evaluation.benchmarks import (
    build_policy_factories,
    collect_matched_episodes,
    compare_all_pairs,
    summarize_episode_results,
)
from cascading_rl.graph.generation import make_ba_graph
from cascading_rl.models import RecoveryQNetwork, build_greedy_policy
from cascading_rl.reproducibility import portable_artifact_path
from scripts.reproducibility import write_run_metadata

POLICY_ORDER = ["rl", "greedy", "degree", "betweenness", "risk", "random"]


def load_checkpoint(path: Path) -> RecoveryQNetwork:
    """
    Load a RecoveryQNetwork checkpoint from the given filesystem path and return the model set to evaluation mode.
    
    Parameters:
        path (Path): Path to a PyTorch checkpoint file containing keys `"model_config"` (for QNetworkConfig) and `"model_state"` (state dict).
    
    Returns:
        RecoveryQNetwork: The instantiated network with weights loaded and .eval() called.
    """
    import torch
    from cascading_rl.models import QNetworkConfig
    data = torch.load(path, map_location="cpu", weights_only=False)
    config = QNetworkConfig(**data["model_config"])
    model = RecoveryQNetwork(config)
    model.load_state_dict(data["model_state"])
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for evaluating a trained recovery policy on larger Barabási–Albert (BA) graphs.
    
    Provides options to specify the model checkpoint, YAML config, target graph sizes to evaluate, number of graphs per size, failure seeds per graph, and the output directory.
    
    Returns:
        argparse.Namespace: Parsed arguments with attributes:
            checkpoint (Path): Path to the model checkpoint.
            config (Path): Path to the YAML configuration file.
            sizes (list[int]): Exact graph sizes to evaluate.
            num_graphs (int): Number of BA graphs to generate per size.
            seeds (list[int]): Failure seeds to use per graph.
            output_dir (Path): Directory where results will be written.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained policy on larger BA graphs (scale generalisation)."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "experiments" / "learner" / "recovery_q.pt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 200, 500,1000],
        help="Exact graph sizes to evaluate (default: 100 200 500, 1000).",
    )
    parser.add_argument(
        "--num-graphs",
        type=int,
        default=20,
        help="Graphs per size (default: 20).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Failure seeds per graph (default: 0..9).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "eval_larger_ba",
    )
    return parser.parse_args()


def _fmt(summary) -> dict:
    """
    Format an episode summary object into a JSON-serializable dictionary with rounded statistics.
    
    Parameters:
        summary: An object containing aggregated episode statistics with attributes
            `anc_fixed`, `final_nc`, `solved_fraction`, `rounds` (each exposing
            `.mean` and for the first two also `.stderr`), and `episode_count`.
    
    Returns:
        A dict with keys:
          - `anc_fixed_mean`: mean of `anc_fixed`, rounded to 4 decimals.
          - `anc_fixed_stderr`: standard error of `anc_fixed`, rounded to 4 decimals.
          - `final_nc_mean`: mean of `final_nc`, rounded to 4 decimals.
          - `final_nc_stderr`: standard error of `final_nc`, rounded to 4 decimals.
          - `solved_fraction_mean`: mean of `solved_fraction`, rounded to 4 decimals.
          - `rounds_mean`: mean of `rounds`, rounded to 2 decimals.
          - `episode_count`: the original episode count (unmodified).
    """
    return {
        "anc_fixed_mean": round(summary.anc_fixed.mean, 4),
        "anc_fixed_stderr": round(summary.anc_fixed.stderr, 4),
        "final_nc_mean": round(summary.final_nc.mean, 4),
        "final_nc_stderr": round(summary.final_nc.stderr, 4),
        "solved_fraction_mean": round(summary.solved_fraction.mean, 4),
        "rounds_mean": round(summary.rounds.mean, 2),
        "episode_count": summary.episode_count,
    }


def run_size(
    n: int,
    *,
    model: RecoveryQNetwork,
    alpha: float,
    pfail: float,
    budget: int,
    max_rounds: int,
    m: int,
    num_graphs: int,
    seeds: list[int],
    scale_budget: bool,
    scale_max_rounds: bool,
    reference_n: int,
) -> tuple[dict, list]:
    """
    Evaluate policies on multiple Barabási–Albert graphs of size n and produce per-policy aggregated summaries and pairwise comparisons against the `degree` baseline.
    
    Parameters:
        n (int): Number of nodes in each generated BA graph.
        model (RecoveryQNetwork): Trained Q-network used to build the RL policy.
        alpha (float): Recovery objective parameter passed to episode collection.
        pfail (float): Per-node failure probability used during evaluation.
        budget (int): Action budget (may be scaled depending on flags).
        max_rounds (int): Maximum rounds per episode (may be scaled depending on flags).
        m (int): Number of edges to attach from a new node in the BA graph generator.
        num_graphs (int): Number of distinct BA graph instances to generate for this size.
        seeds (list[int]): Failure seeds to run per graph (one episode per seed).
        scale_budget (bool): If true, scale `budget` relative to `reference_n`.
        scale_max_rounds (bool): If true, scale `max_rounds` relative to `reference_n`.
        reference_n (int): Reference graph size used when scaling budget or max rounds.
    
    Returns:
        tuple[dict, list]: A pair where the first element is a mapping from policy name to its aggregated summary object (as produced by `summarize_episode_results`), and the second element is a list of pairwise comparison records (comparison vs. the `degree` baseline for the `anc_fixed` metric).
    """
    import torch
    from random import Random

    print(f"\n{'='*55}")
    print(f"Graph size: n = {n}")
    print(f"{'='*55}")

    rng = Random(5000 + n)   # deterministic, separate from training/eval seeds
    graphs = []
    for i in range(num_graphs):
        g = make_ba_graph(n=n, m=m, seed=rng.randint(0, 10**9))
        g.graph["graph_index"] = i
        graphs.append(g)

    avg_deg = sum(2 * g.number_of_edges() / g.number_of_nodes() for g in graphs) / len(graphs)
    print(f"  Graphs: {num_graphs}  n={n}  avg_degree={avg_deg:.2f}")

    device = torch.device("cpu")
    rl_policy = build_greedy_policy(model, device=device, batch_actions=False)
    baseline_factories = build_policy_factories(base_seed=0)
    policy_factories = {
        "rl": lambda gi, se: rl_policy,
        **baseline_factories,
    }

    print(f"  Running {len(policy_factories)} policies × {num_graphs} graphs × {len(seeds)} seeds...", flush=True)
    episodes_by_policy = collect_matched_episodes(
        graphs,
        policy_factories,
        alpha=alpha,
        pfail=pfail,
        budget=budget,
        max_rounds=max_rounds,
        seeds=seeds,
        scale_budget=scale_budget,
        scale_max_rounds=scale_max_rounds,
        reference_n=reference_n,
    )

    summaries = {
        name: summarize_episode_results(eps)
        for name, eps in episodes_by_policy.items()
    }

    comparisons = compare_all_pairs(
        episodes_by_policy,
        baseline="degree",
        metric="anc_fixed",
        rng=__import__("random").Random(0),
    )

    print(f"\n  {'Policy':<14} {'ANC-fixed':>10} {'±stderr':>8} {'Solved':>8} {'Rounds':>7}")
    print(f"  {'-'*50}")
    for name in POLICY_ORDER:
        if name not in summaries:
            continue
        s = summaries[name]
        print(
            f"  {name:<14} {s.anc_fixed.mean:>10.3f} "
            f"{s.anc_fixed.stderr:>8.3f} "
            f"{s.solved_fraction.mean:>8.3f} "
            f"{s.rounds.mean:>7.1f}"
        )

    return summaries, comparisons


def main() -> None:
    """
    Run evaluation of a trained recovery RL policy on larger Barabási–Albert graphs and write summary and metadata files.
    
    Loads configuration and a model checkpoint, evaluates the model alongside baseline policies across the configured graph sizes and seeds, aggregates per-policy summaries and comparisons versus the "degree" baseline, and writes `larger_ba_summary.json` and `run_metadata.json` to the configured output directory.
    
    Notes:
    - Reads evaluation settings from the provided config file and command-line arguments.
    - Writes output summary JSON to `<output_dir>/larger_ba_summary.json`.
    - Writes provenance metadata to `<output_dir>/run_metadata.json`.
    """
    args = parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    training = cfg["training"]
    regime = training["regime"]
    budget_scaling = cfg.get("budget_scaling", {})

    alpha = float(regime["alpha"])
    pfail = float(regime["pfail"])
    budget = int(regime["budget"])
    max_rounds = int(regime["max_rounds"])
    m = int(training["graph"]["m"])
    scale_budget = bool(budget_scaling.get("enabled", True))
    scale_max_rounds = bool(budget_scaling.get("scale_max_rounds", True))
    reference_n = int(budget_scaling.get("reference_n", 40))

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint)

    print(f"\nRegime: alpha={alpha}, pfail={pfail}, budget={budget} (scaled), max_rounds={max_rounds}")
    print(f"Sizes: {args.sizes}  |  num_graphs={args.num_graphs}  |  seeds={args.seeds}")
    print(f"Training range: n ∈ [30, 50]  →  all sizes are OOD")

    results_by_size: dict[str, dict] = {}

    for n in args.sizes:
        summaries, comparisons = run_size(
            n,
            model=model,
            alpha=alpha,
            pfail=pfail,
            budget=budget,
            max_rounds=max_rounds,
            m=m,
            num_graphs=args.num_graphs,
            seeds=args.seeds,
            scale_budget=scale_budget,
            scale_max_rounds=scale_max_rounds,
            reference_n=reference_n,
        )
        results_by_size[str(n)] = {
            "n": n,
            "summaries": {name: _fmt(s) for name, s in summaries.items()},
            "comparisons_vs_degree": [
                {
                    "policy": c.policy_a,
                    "mean_diff_anc_fixed": round(c.mean_difference, 4),
                    "ci_95_low": round(c.bootstrap_ci_low, 4),
                    "ci_95_high": round(c.bootstrap_ci_high, 4),
                    "wilcoxon_p": round(c.wilcoxon_p_value, 4),
                    "significant_p005": c.significant,
                }
                for c in comparisons
            ],
        }

    output = {
        "description": "Scale generalisation: BA graphs at n=100, 200, 500 (all OOD)",
        "training_range": [30, 50],
        "regime": {
            "alpha": alpha,
            "pfail": pfail,
            "budget_ref": budget,
            "budget_scaled": True,
            "reference_n": reference_n,
            "max_rounds": max_rounds,
        },
        "graph_params": {"m": m, "num_graphs_per_size": args.num_graphs, "num_seeds": len(args.seeds)},
        "sizes": results_by_size,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "larger_ba_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved -> {summary_path}")
    write_run_metadata(
        args.output_dir / "run_metadata.json",
        script_path=Path(__file__).resolve(),
        argv=sys.argv,
        config_path=args.config,
        extra={"summary_path": portable_artifact_path(summary_path)},
    )
    print("\nAll done.")


if __name__ == "__main__":
    main()
