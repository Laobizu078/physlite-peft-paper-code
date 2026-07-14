#!/usr/bin/env python3
"""Summarize the paired baseline/trick runs without external statistics packages."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def parse_args() -> argparse.Namespace:
    """Parse result paths and paired-bootstrap settings."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/trick_multiseed"))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--baseline", default="baseline_d")
    parser.add_argument("--candidate", default="query_dominant_lora")
    return parser.parse_args()


def load_runs(root: Path) -> dict[str, dict[int, dict]]:
    """Index result JSON files by method and random seed."""

    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in sorted(root.glob("*/seed_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs[path.parent.name][int(payload["reproducibility"]["seed"])] = payload
    if not runs:
        raise SystemExit(f"No results found below {root}")
    return dict(runs)


def mean_std(values: list[float]) -> str:
    """Format a sample mean and standard deviation for console tables."""

    values_array = np.asarray(values, dtype=np.float64)
    ddof = 1 if len(values) > 1 else 0
    return f"{values_array.mean():.4f} +/- {values_array.std(ddof=ddof):.4f}"


def prediction_map(run: dict) -> dict[str, dict]:
    """Index per-video predictions for paired comparisons."""

    return {row["video_id"]: row for row in run["test"]["predictions"]}


def family_bootstrap(
    baseline: dict,
    candidate: dict,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap candidate-minus-baseline BAcc by complete video family."""

    base = prediction_map(baseline)
    cand = prediction_map(candidate)
    if base.keys() != cand.keys():
        raise ValueError("Paired runs contain different test examples.")

    by_family: dict[str, list[str]] = defaultdict(list)
    for video_id, row in base.items():
        by_family[row["family"]].append(video_id)
    families = sorted(by_family)
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)

    for index in range(samples):
        sampled = rng.choice(families, size=len(families), replace=True)
        video_ids = [video_id for family in sampled for video_id in by_family[family]]
        labels = [base[video_id]["label"] for video_id in video_ids]
        base_pred = [base[video_id]["prediction"] for video_id in video_ids]
        cand_pred = [cand[video_id]["prediction"] for video_id in video_ids]
        differences[index] = balanced_accuracy_score(labels, cand_pred) - balanced_accuracy_score(labels, base_pred)

    low, high = np.quantile(differences, [0.025, 0.975])
    probability_positive = float(np.mean(differences > 0.0))
    return float(low), float(high), probability_positive


def main() -> None:
    """Print multi-seed metrics and paired family-bootstrap intervals."""

    args = parse_args()
    runs = load_runs(args.root)
    print("method                         seeds  test BAcc           test Acc            test F1")
    for method, by_seed in sorted(runs.items()):
        ordered = [by_seed[seed] for seed in sorted(by_seed)]
        print(
            f"{method:30s} {len(ordered):5d}  "
            f"{mean_std([run['test']['balanced_acc'] for run in ordered]):19s} "
            f"{mean_std([run['test']['acc'] for run in ordered]):19s} "
            f"{mean_std([run['test']['f1'] for run in ordered])}"
        )

    baseline_name = args.baseline
    candidate_name = args.candidate
    if baseline_name not in runs or candidate_name not in runs:
        return
    common_seeds = sorted(runs[baseline_name].keys() & runs[candidate_name].keys())
    differences = [
        runs[candidate_name][seed]["test"]["balanced_acc"]
        - runs[baseline_name][seed]["test"]["balanced_acc"]
        for seed in common_seeds
    ]
    print(f"\nPaired BAcc differences ({candidate_name} - {baseline_name}):")
    for seed, difference in zip(common_seeds, differences):
        print(f"  seed {seed}: {difference:+.4f}")
    if differences:
        print(f"  mean:   {np.mean(differences):+.4f}")

    if len(common_seeds) == 3:
        pooled_low = []
        pooled_high = []
        pooled_positive = []
        for seed in common_seeds:
            low, high, positive = family_bootstrap(
                runs[baseline_name][seed],
                runs[candidate_name][seed],
                args.bootstrap_samples,
                args.bootstrap_seed + seed,
            )
            pooled_low.append(low)
            pooled_high.append(high)
            pooled_positive.append(positive)
            print(
                f"  seed {seed} family-bootstrap 95% CI: [{low:+.4f}, {high:+.4f}], "
                f"P(delta>0)={positive:.3f}"
            )
        print(
            "  Note: per-seed family bootstrap measures paired test-set uncertainty; "
            "the three-seed mean/std measures optimization uncertainty."
        )


if __name__ == "__main__":
    main()
