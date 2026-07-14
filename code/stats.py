"""Small statistical helpers used by the paper result aggregator."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict


T_95 = {2: 12.7062, 3: 4.3027, 4: 3.1824, 5: 2.7764}


def mean_std(values: list[float]) -> tuple[float, float]:
    """Return the sample mean and standard deviation."""

    if not values:
        raise ValueError("Cannot summarize an empty sample.")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def paired_interval(left: dict[tuple, float], right: dict[tuple, float]) -> dict:
    """Compute a paired mean difference and two-sided 95% t interval."""

    keys = sorted(left.keys() & right.keys())
    if not keys:
        raise ValueError("The two configurations have no matched runs.")
    differences = [left[key] - right[key] for key in keys]
    mean, std = mean_std(differences)
    half_width = None
    if len(differences) > 1:
        critical = T_95.get(len(differences), 1.96)
        half_width = critical * std / math.sqrt(len(differences))
    return {
        "n": len(keys),
        "keys": [list(key) for key in keys],
        "differences": differences,
        "mean": mean,
        "std": std,
        "ci_low": mean - half_width if half_width is not None else None,
        "ci_high": mean + half_width if half_width is not None else None,
    }


def pareto_front(rows: list[dict], cost_key: str, score_key: str = "test_bacc_mean") -> list[str]:
    """Return configurations not dominated in cost and score."""

    candidates = [row for row in rows if row.get(cost_key) is not None]
    return [
        row["name"]
        for row in candidates
        if not any(
            other[cost_key] <= row[cost_key]
            and other[score_key] >= row[score_key]
            and (other[cost_key] < row[cost_key] or other[score_key] > row[score_key])
            for other in candidates
        )
    ]


def balanced_accuracy(records: list[dict]) -> float:
    """Compute binary balanced accuracy from serialized predictions."""

    recalls = []
    for label in (0, 1):
        selected = [record for record in records if int(record["label"]) == label]
        if not selected:
            raise ValueError("A resample contains only one class.")
        recalls.append(sum(int(record["prediction"]) == label for record in selected) / len(selected))
    return statistics.mean(recalls)


def percentile(values: list[float], quantile: float) -> float:
    """Compute a linearly interpolated empirical percentile."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def family_bootstrap(
    left: dict[int, list[dict]],
    right: dict[int, list[dict]],
    samples: int = 10_000,
    seed: int = 20260712,
    rng: random.Random | None = None,
) -> dict:
    """Bootstrap paired BAcc differences by resampling complete families."""

    seeds = sorted(left.keys() & right.keys())
    if not seeds:
        raise ValueError("No paired prediction records were found.")
    first = left[seeds[0]]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(first):
        groups[str(record["family"])].append(index)
    families = sorted(groups)
    for run_seed in seeds:
        left_ids = [record["video_id"] for record in left[run_seed]]
        right_ids = [record["video_id"] for record in right[run_seed]]
        if left_ids != right_ids or left_ids != [record["video_id"] for record in first]:
            raise ValueError("Prediction order differs across a paired family-bootstrap contrast.")

    observed = [balanced_accuracy(left[s]) - balanced_accuracy(right[s]) for s in seeds]
    rng = rng or random.Random(seed)
    distribution = []
    # Resample families as units so correlated videos never split across draws.
    for _ in range(samples):
        sampled = [rng.choice(families) for _ in families]
        seed_differences = []
        for run_seed in seeds:
            left_sample = [left[run_seed][i] for family in sampled for i in groups[family]]
            right_sample = [right[run_seed][i] for family in sampled for i in groups[family]]
            seed_differences.append(balanced_accuracy(left_sample) - balanced_accuracy(right_sample))
        distribution.append(statistics.mean(seed_differences))
    return {
        "families": len(families),
        "seeds": seeds,
        "observed_by_seed": observed,
        "observed_mean": statistics.mean(observed),
        "samples": samples,
        "ci_low": percentile(distribution, 0.025),
        "ci_high": percentile(distribution, 0.975),
        "positive_probability": sum(value > 0 for value in distribution) / samples,
    }
