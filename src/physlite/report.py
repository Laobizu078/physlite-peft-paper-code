from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from .experiments import ROOT, absolute, build_jobs, load_config
from .models import FrameBackboneClassifier, count_trainable_params
from .stats import family_bootstrap, mean_std, paired_interval, pareto_front
from .train import create_timm_model


MAIN_CONTRASTS = [
    ("pure_support_last8_r2", "composition_lora_r4_lr1e3", "pure LoRA last8/r2 - last4/r4"),
    ("pure_support_last8_r2", "pure_support_last2_r8", "pure LoRA last8/r2 - last2/r8"),
    ("support_last8_r2", "support_last2_r8", "distributed support - concentrated support"),
    ("support_last8_r2", "op_head", "distributed SSF+LoRA - head only"),
    ("allocation_q_last8_r4", "support_last8_r2", "query allocation - equal Q/V allocation"),
    ("allocation_q_last8_r4", "allocation_v_last8_r4", "query allocation - value allocation"),
    ("allocation_q_last8_r4", "rank_r16", "D-SSF-LoRA - late rank-16"),
    ("allocation_q_last8_r4", "op_head", "D-SSF-LoRA - head only"),
    ("allocation_q_last8_r4", "ssf_lora_last4_r4_qv", "D-SSF-LoRA - conventional SSF+LoRA"),
    ("allocation_q_last8_r4", "pure_allocation_q_last8_r4", "D-SSF-LoRA - pure query-LoRA"),
]

SUITE_CONTRASTS = {
    "deit_b": [("d_ssf_lora", "head", "D-SSF-LoRA - head only")],
    "staged": [
        ("joint", "pure", "joint - pure query-LoRA"),
        ("warmup_joint", "joint", "warmup then joint - joint"),
        ("warmup_freeze", "joint", "warmup then freeze - joint"),
    ],
    "repeated": [
        ("distributed", "head", "distributed SSF+LoRA - head only"),
        ("d_ssf_query", "head", "D-SSF-LoRA - head only"),
        ("d_ssf_query", "pure_query", "D-SSF-LoRA - pure query-LoRA"),
    ],
    "prefix8": [("ssf_lora_lr3e4", "head", "prefix PEFT - head only")],
    "prefix16": [("ssf_lora_lr3e4", "head", "prefix PEFT - head only")],
    "backbone_scout": [
        ("deit_peft", "deit_head", "DeiT PEFT - head only"),
        ("dinov2_peft", "dinov2_head", "DINOv2 PEFT - head only"),
    ],
    "readout": [("linear_peft", "linear_head", "linear PEFT - head only")],
}


def run_key(run: dict) -> tuple:
    return (run["manifest"], run["seed"])


def read_run(path: Path, manifest: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    history = data.get("history", [])
    best_epoch = None
    if history:
        best_epoch = max(history, key=lambda row: row["val"]["balanced_acc"])["epoch"]
    scenarios = data.get("test_by_scenario", {})
    return {
        "manifest": manifest.stem,
        "seed": int(data["reproducibility"]["seed"]),
        "val_bacc": data["best_val"]["balanced_acc"],
        "test_bacc": data["test"]["balanced_acc"],
        "test_f1": data["test"]["f1"],
        "trainable_params": data["trainable_params"],
        "trainable_pct": 100.0 * data["trainable_ratio"],
        "peak_mem_mb": data["peak_mem_mb"],
        "seconds": data["seconds"],
        "method": data["method"],
        "best_epoch": best_epoch,
        "scenario_macro_bacc": (
            statistics.mean(row["balanced_acc"] for row in scenarios.values()) if scenarios else None
        ),
        "test_by_scenario": scenarios,
        "predictions": data["test"].get("predictions", []),
    }


def summarize_row(name: str, label: str, axes: list[str], runs: list[dict], head_params: int | None) -> dict:
    metric = lambda key: mean_std([run[key] for run in runs])
    val_mean, val_std = metric("val_bacc")
    bacc_mean, bacc_std = metric("test_bacc")
    f1_mean, f1_std = metric("test_f1")
    trainable = round(statistics.mean(run["trainable_params"] for run in runs))
    row = {
        "name": name,
        "label": label,
        "axes": axes,
        "n": len(runs),
        "method": runs[0]["method"],
        "trainable_params": trainable,
        "peft_params": max(0, trainable - head_params) if head_params is not None else None,
        "trainable_pct": statistics.mean(run["trainable_pct"] for run in runs),
        "val_bacc_mean": val_mean,
        "val_bacc_std": val_std,
        "test_bacc_mean": bacc_mean,
        "test_bacc_std": bacc_std,
        "test_f1_mean": f1_mean,
        "test_f1_std": f1_std,
        "peak_mem_mb_mean": statistics.mean(run["peak_mem_mb"] for run in runs),
        "seconds_mean": statistics.mean(run["seconds"] for run in runs),
        "runs": runs,
    }
    macro = [run["scenario_macro_bacc"] for run in runs if run["scenario_macro_bacc"] is not None]
    if macro:
        row["scenario_macro_bacc_mean"] = statistics.mean(macro)
        row["scenario_macro_bacc_std"] = statistics.stdev(macro) if len(macro) > 1 else 0.0
    return row


def structural_head_params(config: dict, suite: str) -> int:
    settings = {**config["defaults"], **config["protocols"][config["suites"][suite]["protocol"]]}
    backbone = create_timm_model(settings["backbone"], pretrained=False, image_size=settings["image_size"])
    model = FrameBackboneClassifier(
        backbone,
        backbone.num_features,
        temporal=settings["temporal"],
        head_hidden=settings.get("head_hidden", 256),
        head_type=settings.get("head_type", "mlp"),
        freeze_backbone=True,
    )
    trainable, _ = count_trainable_params(model)
    return trainable


def paired(rows: dict[str, dict], contrasts: list[tuple[str, str, str]]) -> list[dict]:
    output = []
    for left, right, label in contrasts:
        if left not in rows or right not in rows:
            continue
        left_values = {run_key(run): run["test_bacc"] for run in rows[left]["runs"]}
        right_values = {run_key(run): run["test_bacc"] for run in rows[right]["runs"]}
        output.append({"left": left, "right": right, "label": label, **paired_interval(left_values, right_values)})
    return output


def scenario_effect(left: dict, right: dict) -> list[dict]:
    left_runs = {run_key(run): run for run in left["runs"]}
    right_runs = {run_key(run): run for run in right["runs"]}
    scenarios = sorted(
        set.intersection(*(set(run["test_by_scenario"]) for run in [*left_runs.values(), *right_runs.values()]))
    )
    output = []
    for scenario in scenarios:
        left_values = {key: run["test_by_scenario"][scenario]["balanced_acc"] for key, run in left_runs.items()}
        right_values = {key: run["test_by_scenario"][scenario]["balanced_acc"] for key, run in right_runs.items()}
        output.append({"scenario": scenario, **paired_interval(left_values, right_values)})
    return output


def main_analysis(rows: dict[str, dict], bootstrap_samples: int) -> dict:
    analysis = {
        "paired_contrasts": paired(rows, MAIN_CONTRASTS),
        "parameter_pareto": pareto_front(list(rows.values()), "peft_params"),
        "memory_pareto": pareto_front(list(rows.values()), "peak_mem_mb_mean"),
    }
    if "allocation_q_last8_r4" in rows and "op_head" in rows:
        analysis["scenario_effect_d_vs_head"] = scenario_effect(rows["allocation_q_last8_r4"], rows["op_head"])
    bootstraps = []
    rng = random.Random(20260712)
    for left, right, label in MAIN_CONTRASTS:
        if left not in rows or right not in rows:
            continue
        left_predictions = {run["seed"]: run["predictions"] for run in rows[left]["runs"]}
        right_predictions = {run["seed"]: run["predictions"] for run in rows[right]["runs"]}
        if all(left_predictions.values()) and all(right_predictions.values()):
            bootstraps.append(
                {
                    "left": left,
                    "right": right,
                    "label": label,
                    **family_bootstrap(left_predictions, right_predictions, bootstrap_samples, rng=rng),
                }
            )
    analysis["family_bootstrap"] = bootstraps
    return analysis


def compact_rows(rows: list[dict]) -> list[dict]:
    keys = ["name", "n", "trainable_params", "peft_params", "test_bacc_mean", "test_bacc_std", "test_f1_mean", "peak_mem_mb_mean"]
    return [{key: row.get(key) for key in keys} for row in rows]


def serializable_rows(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        clean = dict(row)
        clean["runs"] = [
            {key: value for key, value in run.items() if key not in {"predictions", "test_by_scenario"}}
            for run in row["runs"]
        ]
        output.append(clean)
    return output


def markdown(suite: str, rows: list[dict], comparisons: list[dict]) -> str:
    lines = [f"# {suite} results", "", "| Configuration | N | PEFT params | Test BAcc | F1 | Peak MiB |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        peft = "n/a" if row["peft_params"] is None else f"{row['peft_params']:,}"
        lines.append(
            f"| {row['label']} | {row['n']} | {peft} | {row['test_bacc_mean']:.3f} +/- {row['test_bacc_std']:.3f} | "
            f"{row['test_f1_mean']:.3f} | {row['peak_mem_mb_mean']:.0f} |"
        )
    if comparisons:
        lines += ["", "## Paired effects", "", "| Contrast | N | Delta BAcc | 95% t interval |", "| --- | ---: | ---: | ---: |"]
        for item in comparisons:
            interval = "n/a" if item["ci_low"] is None else f"[{100 * item['ci_low']:+.2f}, {100 * item['ci_high']:+.2f}]"
            lines.append(f"| {item['label']} | {item['n']} | {100 * item['mean']:+.2f} | {interval} |")
    return "\n".join(lines) + "\n"


def compare_reference(suite: str, rows: list[dict], reference_dir: Path, tolerance: float) -> list[str]:
    path = reference_dir / f"{suite}.json"
    if not path.exists():
        return [f"{suite}: no reference file"]
    expected = {row["name"]: row for row in json.loads(path.read_text(encoding="utf-8"))}
    messages = []
    for row in rows:
        if row["name"] not in expected:
            messages.append(f"{suite}/{row['name']}: absent from reference")
            continue
        delta = abs(row["test_bacc_mean"] - expected[row["name"]]["test_bacc_mean"])
        status = "PASS" if delta <= tolerance else "FAIL"
        messages.append(f"{status} {suite}/{row['name']}: |delta BAcc|={delta:.6f}")
    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate raw runs and reproduce paper-facing statistics.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper.json")
    parser.add_argument("--suite", action="append", help="Suite to summarize; repeat as needed.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "reference_results")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Compare BAcc means with the released reference results.")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(absolute(args.config))
    output_root = absolute(args.output_root)
    suites = args.suite or list(config["suites"])
    all_payload = {}
    verification = []
    for suite in suites:
        if suite not in config["suites"]:
            raise SystemExit(f"Unknown suite: {suite}")
        jobs = build_jobs(config, suite, output_root, only=None, seeds=None)
        missing = [job.output for job in jobs if not job.output.exists()]
        if missing and not args.allow_incomplete:
            preview = "\n".join(str(path) for path in missing[:5])
            raise FileNotFoundError(f"{suite}: {len(missing)} raw runs are missing. First paths:\n{preview}")

        run_spec = {run["name"]: run for run in config["suites"][suite]["runs"]}
        grouped: dict[str, list[dict]] = {name: [] for name in run_spec}
        for job in jobs:
            if job.output.exists():
                grouped[job.name].append(read_run(job.output, job.manifest))
        head_name = next((name for name in ("op_head", "head", "deit_head", "linear_head") if grouped.get(name)), None)
        head_params = None
        if head_name:
            head_params = round(statistics.mean(run["trainable_params"] for run in grouped[head_name]))
        else:
            head_params = structural_head_params(config, suite)
        rows = {
            name: summarize_row(name, run_spec[name]["label"], run_spec[name].get("axes", []), runs, head_params)
            for name, runs in grouped.items()
            if runs
        }
        comparisons = paired(rows, SUITE_CONTRASTS.get(suite, []))
        payload = {
            "suite": suite,
            "expected_jobs": len(jobs),
            "completed_jobs": len(jobs) - len(missing),
            "rows": serializable_rows(list(rows.values())),
            "paired_contrasts": comparisons,
        }
        if suite == "main":
            payload["analysis"] = main_analysis(rows, args.bootstrap_samples)
        suite_dir = output_root / suite
        suite_dir.mkdir(parents=True, exist_ok=True)
        (suite_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (suite_dir / "summary.md").write_text(markdown(suite, list(rows.values()), comparisons), encoding="utf-8")
        all_payload[suite] = {"rows": compact_rows(list(rows.values())), "paired_contrasts": comparisons}
        if args.verify:
            verification.extend(compare_reference(suite, list(rows.values()), absolute(args.reference_dir), args.tolerance))
        print(f"{suite}: summarized {len(jobs) - len(missing)}/{len(jobs)} runs")

    (output_root / "paper_results.json").write_text(json.dumps(all_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.verify:
        print("\n".join(verification))
        failures = [message for message in verification if message.startswith("FAIL")]
        if failures:
            raise SystemExit(f"Reference verification failed for {len(failures)} configurations.")


if __name__ == "__main__":
    main()
