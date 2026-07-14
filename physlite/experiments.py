"""Expand the paper configuration into validated, reproducible training jobs."""

from __future__ import annotations

import argparse
import glob
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Job:
    """A fully resolved training run with its manifest and output locations."""

    suite: str
    name: str
    label: str
    seed: int
    manifest: Path
    output: Path
    checkpoint: Path | None
    arguments: dict


def load_config(path: Path) -> dict:
    """Load a versioned paper experiment configuration from JSON."""

    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported config schema in {path}")
    return config


def absolute(path: str | Path) -> Path:
    """Resolve repository-relative paths while preserving absolute paths."""

    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def manifests_for(config: dict, suite: dict) -> list[Path]:
    """Resolve the manifest files selected by an experiment suite."""

    pattern = suite.get("manifests")
    if pattern:
        paths = [Path(path) for path in sorted(glob.glob(str(absolute(pattern))))]
        if not paths:
            raise RuntimeError(f"No manifests matched {pattern}")
        return paths
    return [absolute(config["defaults"]["manifest"])]


def build_jobs(config: dict, suite_name: str, output_root: Path, only: set[str] | None, seeds: list[int] | None) -> list[Job]:
    """Expand one suite across configurations, manifests, and random seeds."""

    suite = config["suites"][suite_name]
    protocol = config["protocols"][suite["protocol"]]
    manifests = manifests_for(config, suite)
    jobs = []
    for run in suite["runs"]:
        if only and run["name"] not in only:
            continue
        method = config["methods"][run["method"]]
        arguments = {**config["defaults"], **protocol, **method, **run.get("overrides", {})}
        run_seeds = seeds if seeds is not None else run.get("seeds", suite["seeds"])
        for manifest in manifests:
            for seed in run_seeds:
                relative = Path(suite_name) / run["name"]
                if len(manifests) > 1:
                    relative /= manifest.stem
                output = output_root / relative / f"seed_{seed}.json"
                checkpoint = output.with_suffix(".pt") if run.get("checkpoint", False) else None
                jobs.append(
                    Job(
                        suite=suite_name,
                        name=run["name"],
                        label=run["label"],
                        seed=seed,
                        manifest=manifest,
                        output=output,
                        checkpoint=checkpoint,
                        arguments=arguments,
                    )
                )
    return jobs


def command(job: Job) -> list[str]:
    """Build the interpreter-safe command for one training job."""

    arguments = dict(job.arguments)
    arguments["manifest"] = str(job.manifest)
    arguments["data_root"] = str(absolute(arguments["data_root"]))
    arguments["seed"] = job.seed
    arguments["output_json"] = str(job.output)
    if job.checkpoint is not None:
        arguments["checkpoint_out"] = str(job.checkpoint)

    cmd = [sys.executable, "-m", "physlite.train"]
    for key, value in arguments.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif value is not None:
            cmd.extend([flag, str(value)])
    return cmd


def validate_jobs(jobs: list[Job]) -> None:
    """Reject duplicate outputs, missing manifests, and non-CUDA paper jobs."""

    outputs = [job.output for job in jobs]
    if len(outputs) != len(set(outputs)):
        raise RuntimeError("Experiment config generates duplicate output paths.")
    for job in jobs:
        if not job.manifest.is_file():
            raise FileNotFoundError(job.manifest)
        if job.arguments.get("device") != "cuda":
            raise RuntimeError(f"Paper jobs must force CUDA: {job.suite}/{job.name}")


def parse_args() -> argparse.Namespace:
    """Parse experiment-suite selection and execution controls."""

    parser = argparse.ArgumentParser(description="Run one or more paper experiment suites sequentially on one GPU.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper.json")
    parser.add_argument("--suite", action="append", help="Suite name; repeat for multiple suites.")
    parser.add_argument("--only", nargs="+", help="Run only these configuration names.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Override configured seeds.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--force", action="store_true", help="Rerun jobs whose JSON result already exists.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run or list the resolved paper jobs in deterministic sequence."""

    args = parse_args()
    config = load_config(absolute(args.config))
    if args.list:
        for name, suite in config["suites"].items():
            print(f"{name:18s} {len(suite['runs']):2d} configurations")
        return
    suites = args.suite or list(config["suites"])
    unknown = set(suites) - set(config["suites"])
    if unknown:
        raise SystemExit(f"Unknown suites: {', '.join(sorted(unknown))}")
    jobs = []
    for suite in suites:
        jobs.extend(build_jobs(config, suite, absolute(args.output_root), set(args.only or []) or None, args.seeds))
    validate_jobs(jobs)
    if not args.dry_run and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to fall back to CPU.")

    completed = 0
    for index, job in enumerate(jobs, start=1):
        cmd = command(job)
        cached = job.output.exists() and (job.checkpoint is None or job.checkpoint.exists())
        if cached and not args.force:
            print(f"[{index}/{len(jobs)}] SKIP {job.suite}/{job.name} seed={job.seed}")
            completed += 1
            continue
        print(f"[{index}/{len(jobs)}] RUN  {job.suite}/{job.name} seed={job.seed}")
        print(shlex.join(cmd))
        if args.dry_run:
            continue
        job.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        completed += 1
    print(f"jobs={len(jobs)} completed_or_cached={completed} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
