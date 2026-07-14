#!/usr/bin/env python3
"""Summarize validation performance and EMA selection frequency."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    """Print mean validation BAcc and how often EMA won selection."""

    rows = defaultdict(list)
    for path in sorted(Path("outputs/ema_scout").glob("*/seed_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows[path.parent.name].append(payload)

    print("configuration       N  val BAcc mean +/- std  EMA selections")
    for name, runs in sorted(rows.items()):
        values = np.asarray([run["best_val"]["balanced_acc"] for run in runs])
        selections = sum(run["best_variant"] == "ema" for run in runs)
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        print(f"{name:18s} {len(runs):2d}  {values.mean():.4f} +/- {std:.4f}       {selections}/{len(runs)}")


if __name__ == "__main__":
    main()
