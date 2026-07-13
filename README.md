# PhysLite-PEFT

Code release for the controlled study of parameter-efficient fine-tuning (PEFT)
on physical video understanding. The repository reproduces every quantitative
result used in the course paper from a single experiment specification.

The central experiment freezes an ImageNet-pretrained DeiT frame encoder and
systematically varies **what is tuned, where it is inserted, how the parameter
budget is allocated, and how it is optimized**. Physion is the downstream task;
video frames are encoded independently and an order-sensitive lightweight head
aggregates their features.

## Quick start

The released environment targets one NVIDIA GPU and was tested on an RTX 4070 Ti
SUPER (16 GB). Paper runs deliberately request CUDA and never silently use CPU.

```bash
conda env create -f environment.yml
conda activate cvpr_paper
pip install -e . --no-deps

# Download the 271 MiB archive, verify its SHA-256, extract it, and audit splits.
physlite-prepare

# Unit checks do not download pretrained weights or require a GPU.
pytest -q
```

For an existing Physion extraction, place or link it at `data/Physion`, then run
`physlite-prepare --skip-download`. The CSV manifests are tracked and contain only
paths relative to that directory.

## Reproduce results

List the available suites:

```bash
physlite-run --list
```

Run and aggregate the 84-run main design matrix:

```bash
physlite-run --suite main
physlite-report --suite main --verify
```

Run one configuration or inspect exact commands without training:

```bash
physlite-run --suite main --only allocation_q_last8_r4
physlite-run --suite main --only allocation_q_last8_r4 --dry-run
```

`physlite-run` resumes at the granularity of one JSON file. Add `--force` to
replace completed runs. Every raw result records the command, seed, manifest
checksum, package versions, CUDA version, GPU, predictions, and per-scenario
metrics. Aggregation writes `summary.json`, `summary.md`, and a combined
`outputs/paper_results.json`.

The complete release can be reproduced with:

```bash
bash scripts/reproduce_all.sh
```

That script executes 168 training jobs, then evaluates the saved rank-8 models
under six temporal counterfactuals. It is intentionally sequential for a single
16 GB GPU. `scripts/reproduce_core.sh` runs the main PEFT evidence chain only.

## Experiment suites

| Suite | Jobs | Purpose |
| --- | ---: | --- |
| `main` | 84 | 28-configuration PEFT design matrix, three seeds |
| `deit_b` | 18 | Width transfer from DeiT-S to DeiT-B |
| `staged` | 12 | Joint, warm-up, and freeze schedules |
| `repeated` | 20 | Four methods on five family-grouped splits |
| `prefix8` | 11 | 1.5 s observed-prefix study, eight frames |
| `prefix16` | 6 | Higher temporal sampling at the same horizon |
| `backbone_scout` | 12 | High-resolution DeiT and DINOv2 controls |
| `counterfactual` | 3 | Checkpointed rank-8 motion-statistics model |
| `readout` | 2 | Linear temporal readout control |

The exact mapping from paper tables to suites is in
[`docs/RESULTS_MAP.md`](docs/RESULTS_MAP.md). All hyperparameters are centralized
in [`configs/paper.json`](configs/paper.json); method implementation lives in
[`src/physlite/models.py`](src/physlite/models.py), while the runner contains no
method-specific branches.

## Released references

`reference_results/` contains compact snapshots generated from the original raw
runs. They provide a regression target without shipping checkpoints or duplicate
video predictions. Verification compares mean balanced accuracy with a default
tolerance of 0.02, accommodating ordinary GPU-level numerical variation:

```bash
physlite-report --verify
```

The main result was additionally regression-tested by rebuilding all statistics,
including 10,000-sample family-level bootstraps, from its 84 original JSON files.

## Repository layout

```text
configs/paper.json          all protocols, methods, suites, and seeds
data/manifests/             fixed leakage-free main and repeated splits
reference_results/          compact published metric snapshots
src/physlite/models.py      PEFT operators and video classifier
src/physlite/train.py       one-run training/evaluation kernel
src/physlite/experiments.py config expansion, CUDA enforcement, resume
src/physlite/report.py      aggregation and statistical analysis
src/physlite/probe.py       temporal counterfactual evaluation
tests/                      protocol, portability, and parameter-budget checks
```

## Data and weights

`physlite-prepare` downloads the public Physion archive from its official NeurIPS
2021 benchmark endpoint and checks SHA-256
`1c80e51d9d299a54cc78bb20b9bb9b597d3b18067fd2f5a06e4e0a3a0c2c0c26`.
Pretrained DeiT/DINOv2 weights are fetched by `timm` on first use. Dataset and
pretrained-model licenses remain those of their respective authors.
