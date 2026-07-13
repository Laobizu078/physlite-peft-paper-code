#!/usr/bin/env python3
"""One-time provenance bridge from the development tree to release references."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


MAPPINGS = {
    "main": (
        "peft_design_matrix",
        {},
    ),
    "deit_b": (
        "motion_bins_gap",
        {
            "deit_base_head_motion_bins": "head",
            "deit_base_ssf_lora_last2_r8_motion_bins": "last2_r8",
            "deit_base_ssf_lora_r4_motion_bins": "last4_r4",
            "deit_base_ssf_lora_last8_r2_motion_bins": "last8_r2",
            "deit_base_lora_last8_q_r4_motion_bins": "pure_query",
            "deit_base_d_ssf_lora_last8_q_r4_motion_bins": "d_ssf_lora",
        },
    ),
    "staged": ("staged_ssf_lora_q_r4", {}),
    "prefix8": (
        "observed_prefix_study",
        {
            "head_motion_bins": "head",
            "ssf_motion_bins": "ssf",
            "lora_motion_bins": "lora",
            "ssf_lora_motion_bins": "ssf_lora",
            "ssf_lora_lr3e4_motion_bins": "ssf_lora_lr3e4",
            "temporal_lora_motion_bins": "temporal_lora",
            "ssf_temporal_lora_motion_bins": "ssf_temporal_lora",
        },
    ),
    "prefix16": (
        "observed_prefix_16",
        {"head_motion_bins": "head", "ssf_lora_lr3e4_motion_bins": "ssf_lora_lr3e4"},
    ),
    "backbone_scout": (
        "highres_dinov2_scout",
        {
            "deit_small_head_f8_512_b4": "deit_head",
            "deit_small_rank8_f8_512_b4": "deit_peft",
            "dinov2_small_head_f8_518_b4": "dinov2_head",
            "dinov2_small_rank8_f8_518_b4": "dinov2_peft",
        },
    ),
    "readout": (
        "observed_prefix_linear",
        {"head_motion_bins": "linear_head", "ssf_lora_lr3e4_motion_bins": "linear_peft"},
    ),
}


KEEP = [
    "n",
    "trainable_params",
    "peft_params",
    "trainable_pct",
    "val_bacc_mean",
    "val_bacc_std",
    "test_bacc_mean",
    "test_bacc_std",
    "test_f1_mean",
    "test_f1_std",
    "peak_mem_mb_mean",
    "seconds_mean",
]


def compact(row: dict, name: str) -> dict:
    return {"name": name, **{key: row[key] for key in KEEP if key in row}}


def import_repeated(root: Path) -> list[dict]:
    payload = json.loads((root / "repeated_split_budget_validation/summary.json").read_text(encoding="utf-8"))
    by_config: dict[str, list[float]] = {}
    for row in payload["rows"]:
        by_config.setdefault(row["config"], []).append(row["test_bacc_mean"])
    return [
        {
            "name": name,
            "n": len(values),
            "test_bacc_mean": statistics.mean(values),
            "test_bacc_std": statistics.stdev(values),
        }
        for name, values in by_config.items()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_outputs", type=Path)
    parser.add_argument("--destination", type=Path, default=Path("reference_results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    for suite, (directory, names) in MAPPINGS.items():
        rows = json.loads((args.legacy_outputs / directory / "summary.json").read_text(encoding="utf-8"))
        selected = []
        for row in rows:
            old_name = row["name"]
            if names and old_name not in names:
                continue
            selected.append(compact(row, names.get(old_name, old_name)))
        (args.destination / f"{suite}.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")

    repeated = import_repeated(args.legacy_outputs)
    (args.destination / "repeated.json").write_text(json.dumps(repeated, indent=2) + "\n", encoding="utf-8")

    shortcut = json.loads((args.legacy_outputs / "rank8_shortcut_analysis/summary.json").read_text(encoding="utf-8"))
    counterfactual = [compact(shortcut["train_controls"][0], "rank8_motion_stats")]
    (args.destination / "counterfactual.json").write_text(json.dumps(counterfactual, indent=2) + "\n", encoding="utf-8")
    probes = [{key: row[key] for key in ("mode", "n", "bacc_mean", "bacc_std", "drop_from_original")} for row in shortcut["counterfactual"]]
    (args.destination / "counterfactual_probes.json").write_text(json.dumps(probes, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
