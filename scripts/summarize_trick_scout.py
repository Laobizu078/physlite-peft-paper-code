#!/usr/bin/env python3
"""Rank validation-only trick scouts and verify saved best epochs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Print the scout configurations ordered by validation BAcc."""

    rows = []
    for path in sorted((ROOT / "outputs/trick_scout").glob("*/seed_0.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            (
                path.parent.name,
                result["best_val"]["balanced_acc"],
                result["best_val"]["f1"],
                max(row["val"]["balanced_acc"] for row in result["history"]),
                result["peak_mem_mb"],
            )
        )
    rows.sort(key=lambda row: row[1], reverse=True)
    print("configuration                best val BAcc  val F1  peak MiB")
    for name, bacc, f1, history_best, memory in rows:
        if abs(bacc - history_best) > 1e-12:
            raise RuntimeError(f"Best validation mismatch for {name}")
        print(f"{name:28s} {bacc:13.4f}  {f1:6.4f}  {memory:8.0f}")


if __name__ == "__main__":
    main()
