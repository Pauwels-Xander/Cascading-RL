"""Topology ablation: BA vs ER vs WS at matched scale and degree.

Tier 1 of the two-tier evaluation structure:

    Tier 1 — Topology ablation (n ∈ [30, 50], matched average degree)
        BA  : Barabási-Albert, scale-free, m=2 → avg degree ≈ 4  (training distribution)
        ER  : Erdős-Rényi,     random,     p = 2m/n → avg degree ≈ 4
        WS  : Watts-Strogatz,  small-world, k=4, p=0.1 → avg degree = 4

    Tier 2 — OOD evaluation (n = 300, real/realistic topologies)
        See scripts/evaluate_real_world.py

All three graph types use the same n_range, m, and failure regime drawn from
config/default.yaml. Graphs are generated with distinct seeds so BA/ER/WS sets
do not overlap; the episode seed lists are identical across types so comparisons
are matched on failure scenario.

Output
------
experiments/eval_topology_ablation/topology_ablation_summary.json
experiments/eval_topology_ablation/run_metadata.json

Usage
-----
    python scripts/evaluate_topology_ablation.py
    python scripts/evaluate_topology_ablation.py --num-graphs 50 --seeds 0 1 2 3 4
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
from cascading_rl.graph.generation import make_graph_batch
from cascading_rl.models import RecoveryQNetwork, build_greedy_policy
from cascading_rl.reproducibility import portable_artifact_path
from scripts.reproducibility import write_run_metadata

# Distinct graph-generation seeds for each topology type so the graph pools
# do not overlap.  Episode failure seeds are shared across types.
_GRAPH_SEEDS = {"ba": 0, "er": 999, "ws": 1999}

POLICY_ORDER = ["rl", "greedy", "degree", "betweenness", "risk", "random"]


def load_checkpoint(path: Path) -> RecoveryQNetwork:
    """
    Load a RecoveryQNetwork from a PyTorch checkpoint file.
    
    Parameters:
        path (Path): Filesystem path to a PyTorch checkpoint containing the keys
            "model_config" (mapping of QNetworkConfig parameters) and "model_state"
            (state_dict for the model).
    
    Returns:
        RecoveryQNetwork: The reconstructed model set to evaluation mode.
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
    Parse command-line arguments for the topology ablation evaluation.
    
    @returns:
        argparse.Namespace: Parsed arguments with the following attributes:
            checkpoint (Path): Path to the model checkpoint.
            config (Path): Path to the YAML configuration file.
            topologies (list[str]): Topology types to include (e.g., ["ba","er","ws"]).
            num_graphs (int): Number of graphs to generate per topology type.
            seeds (list[int]): Failure/episode seeds to use per graph.
            output_dir (Path): Directory where outputs and metadata will be written.
    """
    parser = argparse.ArgumentParser(
        description="Topology ablation: BA vs ER vs WS at n∈[30,50], matched degree."
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
        "--topologies",
        nargs="+",
        default=["ba", "er", "ws"],
        help="Which topology types to include (default: ba er ws).",
    )
    parser.add_argument(
        "--num-graphs",
        type=int,
        default=100,
        help="Number of graphs per topology type (default: 100).",
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
        default=ROOT / "experiments" / "eval_topology_ablation",
    )
    return parser.parse_args()


def _fmt_summary(summary) -> dict:
    """
    Format a summary object into a plain dictionary with selected, rounded statistics.
    
    Parameters:
        summary: An object with the following attributes:
            - anc_fixed: object with `mean` and `stderr` numeric attributes
            - final_nc: object with `mean` and `stderr` numeric attributes
            - solved_fraction: object with a `mean` numeric attribute
            - rounds: object with a `mean` numeric attribute
            - episode_count: integer count of episodes
    
    Returns:
        dict: A mapping with these keys:
            - "anc_fixed_mean": `anc_fixed.mean` rounded to 4 decimal places
            - "anc_fixed_stderr": `anc_fixed.stderr` rounded to 4 decimal places
            - "final_nc_mean": `final_nc.mean` rounded to 4 decimal places
            - "final_nc_stderr": `final_nc.stderr` rounded to 4 decimal places
            - "solved_fraction_mean": `solved_fraction.mean` rounded to 4 decimal places
            - "rounds_mean": `rounds.mean` rounded to 2 decimal places
            - "episode_count": the original `episode_count` value
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


def run_topology(
    topology: str,
    *,
    model: RecoveryQNetwork,
    alpha: float,
    pfail: float,
    budget: int,
    max_rounds: int,
    m: int,
    n_range: tuple[int, int],
    num_graphs: int,
    seeds: list[int],
    scale_budget: bool,
    scale_max_rounds: bool,
    reference_n: int,
) -> tuple[dict, list]:
    """
    Run evaluation of all policies on a single graph topology and produce per-policy summaries and pairwise comparisons.
    
    Parameters:
        topology (str): Topology name (e.g., "ba", "er", "ws") to generate graphs for.
        model (RecoveryQNetwork): Loaded recovery Q-network used to build the RL greedy policy.
        alpha (float): Regime parameter controlling attack/repair trade-off.
        pfail (float): Per-episode failure probability used by the environment.
        budget (int): Action budget per episode (may be scaled by `scale_budget`).
        max_rounds (int): Maximum rounds per episode (may be scaled by `scale_max_rounds`).
        m (int): Graph model parameter controlling target average degree (≈ 2*m for BA/WS).
        n_range (tuple[int, int]): Inclusive range of graph sizes (number of nodes) to sample.
        num_graphs (int): Number of graph instances to generate for this topology.
        seeds (list[int]): List of episode/failure seeds shared across topologies to align failure regimes.
        scale_budget (bool): If True, scale `budget` according to `reference_n` and sampled graph size.
        scale_max_rounds (bool): If True, scale `max_rounds` according to `reference_n` and sampled graph size.
        reference_n (int): Reference number of nodes used when applying budget/round scaling.
    
    Returns:
        tuple[dict, list]: A pair where the first element is a mapping from policy name to its summarized metrics,
        and the second element is a list of pairwise comparison results (each comparing a policy to the "degree" baseline
        for the `anc_fixed` metric).
    """
    import torch
    print(f"\n{'='*55}")
    print(f"Topology: {topology.upper()}")
    print(f"{'='*55}")

    graph_seed = _GRAPH_SEEDS.get(topology, hash(topology) % 10**6)
    graphs = make_graph_batch(
        num_graphs=num_graphs,
        n_range=n_range,
        m=m,
        seed=graph_seed,
        graph_type=topology,
    )
    avg_n = sum(g.number_of_nodes() for g in graphs) / len(graphs)
    avg_deg = sum(2 * g.number_of_edges() / g.number_of_nodes() for g in graphs) / len(graphs)
    print(f"  Graphs: {len(graphs)}  avg_n={avg_n:.1f}  avg_degree={avg_deg:.2f}")

    device = torch.device("cpu")
    rl_policy = build_greedy_policy(model, device=device, batch_actions=False)
    baseline_factories = build_policy_factories(base_seed=0)
    policy_factories = {
        "rl": lambda gi, se: rl_policy,
        **baseline_factories,
    }

    print(f"  Running {len(policy_factories)} policies × {len(graphs)} graphs × {len(seeds)} seeds...", flush=True)
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

    # Print table
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
    Run the topology ablation experiment and persist a JSON summary and run metadata.
    
    Parses command-line arguments, loads the evaluation configuration and a trained recovery Q-network checkpoint, evaluates the configured policies across the requested graph topologies (BA, ER, WS) using the specified regime and graph parameters, aggregates per-policy summaries and comparisons versus the degree baseline, and writes a summary JSON and run metadata file into the output directory.
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
    n_range = tuple(training["graph"]["n_range"])
    scale_budget = bool(budget_scaling.get("enabled", True))
    scale_max_rounds = bool(budget_scaling.get("scale_max_rounds", True))
    reference_n = int(budget_scaling.get("reference_n", 40))

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint)

    print(f"\nRegime: alpha={alpha}, pfail={pfail}, budget={budget}, max_rounds={max_rounds}")
    print(f"Graph params: n_range={n_range}, m={m} (avg degree ≈ {2*m})")
    print(f"Topologies: {args.topologies}  |  num_graphs={args.num_graphs}  |  seeds={args.seeds}")

    results_by_topology: dict[str, dict] = {}

    for topology in args.topologies:
        summaries, comparisons = run_topology(
            topology,
            model=model,
            alpha=alpha,
            pfail=pfail,
            budget=budget,
            max_rounds=max_rounds,
            m=m,
            n_range=n_range,
            num_graphs=args.num_graphs,
            seeds=args.seeds,
            scale_budget=scale_budget,
            scale_max_rounds=scale_max_rounds,
            reference_n=reference_n,
        )
        results_by_topology[topology] = {
            "summaries": {name: _fmt_summary(s) for name, s in summaries.items()},
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
        "tier": "topology_ablation",
        "description": "BA vs ER vs WS at n∈[30,50] with matched average degree (~4)",
        "regime": {
            "alpha": alpha,
            "pfail": pfail,
            "budget": budget,
            "max_rounds": max_rounds,
        },
        "graph_params": {
            "n_range": list(n_range),
            "m": m,
            "avg_degree_target": 2 * m,
            "ws_k": 2 * m,
            "ws_p": 0.1,
            "er_p_formula": "2*m/n",
            "num_graphs_per_topology": args.num_graphs,
            "num_seeds": len(args.seeds),
        },
        "topologies": results_by_topology,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "topology_ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nSaved -> {summary_path}")

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
