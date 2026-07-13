from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import timm

from physlite.datasets import select_frame_indices
from physlite.experiments import build_jobs, command, load_config
from physlite.models import FrameBackboneClassifier, apply_lora_qv_to_vit, apply_ssf_to_layernorm, count_trainable_params
from physlite.train import parse_args


ROOT = Path(__file__).resolve().parents[1]


def test_observed_sampling_never_crosses_horizon() -> None:
    indices = select_frame_indices(total=602, fps=30.0, frames=8, sampling="observed", observation_seconds=1.5)
    assert len(indices) == 8
    assert int(indices[0]) == 0
    assert int(indices[-1]) == 44
    assert int(indices.max()) < 45


def test_uniform_sampling_reaches_outcome_frame() -> None:
    indices = select_frame_indices(total=151, fps=30.0, frames=8, sampling="uniform")
    assert int(indices[-1]) == 150


def test_repeated_manifests_have_no_family_leakage() -> None:
    paths = sorted((ROOT / "data" / "manifests" / "repeated_splits").glob("split_*.csv"))
    assert len(paths) == 5
    for path in paths:
        frame = pd.read_csv(path)
        assert len(frame) == 1200
        assert int((frame.groupby("family")["split"].nunique() > 1).sum()) == 0
        assert set(frame["split"]) == {"pilot_train", "pilot_val", "pilot_test"}


def test_release_manifests_are_portable() -> None:
    paths = [ROOT / "data/manifests/main.csv", *(ROOT / "data/manifests/repeated_splits").glob("split_*.csv")]
    assert len(paths) == 6
    for path in paths:
        frame = pd.read_csv(path)
        assert len(frame) == 1200
        assert not frame["video_path"].map(Path).map(Path.is_absolute).any()


def test_config_expands_to_all_168_paper_jobs(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/paper.json")
    expected = {
        "main": 84,
        "deit_b": 18,
        "staged": 12,
        "repeated": 20,
        "prefix8": 11,
        "prefix16": 6,
        "backbone_scout": 12,
        "counterfactual": 3,
        "readout": 2,
    }
    counts = {name: len(build_jobs(config, name, tmp_path, None, None)) for name in config["suites"]}
    assert counts == expected
    assert sum(counts.values()) == 168


def test_every_config_has_a_released_reference() -> None:
    config = load_config(ROOT / "configs/paper.json")
    for suite, specification in config["suites"].items():
        configured = {run["name"] for run in specification["runs"]}
        released = {
            row["name"]
            for row in json.loads((ROOT / "reference_results" / f"{suite}.json").read_text(encoding="utf-8"))
        }
        assert configured == released


def test_all_generated_commands_are_accepted_by_training_cli(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/paper.json")
    for suite in config["suites"]:
        for job in build_jobs(config, suite, tmp_path, None, None):
            argv = ["physlite-train", *command(job)[3:]]
            with patch("sys.argv", argv):
                parsed = parse_args()
            assert parsed.device == "cuda"


def test_reported_parameter_budget_is_structurally_reproducible() -> None:
    backbone = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=0)
    model = FrameBackboneClassifier(
        backbone,
        backbone.num_features,
        temporal="motion_bins",
        head_hidden=256,
        freeze_backbone=True,
    )
    head_params, _ = count_trainable_params(model)
    apply_ssf_to_layernorm(model.backbone, layers="all")
    apply_lora_qv_to_vit(model.backbone, rank=4, alpha=8, layers="last8", lora_targets="q")
    d_params, _ = count_trainable_params(model)
    assert head_params == 694_274
    assert d_params - head_params == 43_776
