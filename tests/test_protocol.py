"""Regression tests for protocol isolation, job expansion, and PEFT tricks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import timm
import torch

from physlite.datasets import select_frame_indices
from physlite.experiments import build_jobs, command, load_config
from physlite.models import (
    FrameBackboneClassifier,
    apply_lora_qv_to_vit,
    apply_query_dominant_lora_to_vit,
    apply_ssf_to_layernorm,
    count_trainable_params,
)
from physlite.train import TrainableEMA, apply_video_hflip, build_optimizer_groups, parse_args, update_optimizer_lrs


ROOT = Path(__file__).resolve().parents[1]


def test_observed_sampling_never_crosses_horizon() -> None:
    """Observed-prefix sampling must not access post-horizon frames."""

    indices = select_frame_indices(total=602, fps=30.0, frames=8, sampling="observed", observation_seconds=1.5)
    assert len(indices) == 8
    assert int(indices[0]) == 0
    assert int(indices[-1]) == 44
    assert int(indices.max()) < 45


def test_uniform_sampling_reaches_outcome_frame() -> None:
    """Full-video sampling must include the final outcome frame."""

    indices = select_frame_indices(total=151, fps=30.0, frames=8, sampling="uniform")
    assert int(indices[-1]) == 150


def test_repeated_manifests_have_no_family_leakage() -> None:
    """Every repeated split must keep each family in one partition."""

    paths = sorted((ROOT / "data" / "manifests" / "repeated_splits").glob("split_*.csv"))
    assert len(paths) == 5
    for path in paths:
        frame = pd.read_csv(path)
        assert len(frame) == 1200
        assert int((frame.groupby("family")["split"].nunique() > 1).sum()) == 0
        assert set(frame["split"]) == {"pilot_train", "pilot_val", "pilot_test"}


def test_release_manifests_are_portable() -> None:
    """Released manifests must use repository-relative video paths."""

    paths = [ROOT / "data/manifests/main.csv", *(ROOT / "data/manifests/repeated_splits").glob("split_*.csv")]
    assert len(paths) == 6
    for path in paths:
        frame = pd.read_csv(path)
        assert len(frame) == 1200
        assert not frame["video_path"].map(Path).map(Path.is_absolute).any()


def test_config_expands_to_all_168_paper_jobs(tmp_path: Path) -> None:
    """The release config must expand to the paper's complete job matrix."""

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
    """Every configured experiment must have a matching reference row."""

    config = load_config(ROOT / "configs/paper.json")
    for suite, specification in config["suites"].items():
        configured = {run["name"] for run in specification["runs"]}
        released = {
            row["name"]
            for row in json.loads((ROOT / "reference_results" / f"{suite}.json").read_text(encoding="utf-8"))
        }
        assert configured == released


def test_all_generated_commands_are_accepted_by_training_cli(tmp_path: Path) -> None:
    """Generated commands must remain compatible with the training parser."""

    config = load_config(ROOT / "configs/paper.json")
    for suite in config["suites"]:
        for job in build_jobs(config, suite, tmp_path, None, None):
            argv = ["physlite-train", *command(job)[3:]]
            with patch("sys.argv", argv):
                parsed = parse_args()
            assert parsed.device == "cuda"


def test_reported_parameter_budget_is_structurally_reproducible() -> None:
    """D-SSF-LoRA's reported net parameter budget must follow from the model."""

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


def test_depth_focused_loraplus_preserves_parameters_and_expected_lrs() -> None:
    """Depth-focused LoRA+ must only redistribute learning rates."""

    backbone = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=0)
    model = FrameBackboneClassifier(
        backbone,
        backbone.num_features,
        temporal="motion_bins",
        head_hidden=256,
        freeze_backbone=True,
    )
    apply_ssf_to_layernorm(model.backbone, layers="all")
    apply_lora_qv_to_vit(model.backbone, rank=4, alpha=8, layers="last8", lora_targets="q")
    args = SimpleNamespace(
        lr=1e-3,
        weight_decay=0.05,
        ssf_lr=2e-4,
        peft_lr=1e-3,
        lora_plus_ratio=4.0,
        lora_depth_profile="middle_focus",
        lora_depth_strength=0.5,
        lora_allocation_warmup_epochs=2,
        ssf_warmup_epochs=0,
        freeze_ssf_after_warmup=False,
    )
    groups = build_optimizer_groups(model, args)
    lrs = {group["group_name"]: group["lr"] for group in groups}
    assert lrs == {
        "head": 1e-3,
        "ssf": 2e-4,
        "lora_a_middle": 7.5e-4,
        "lora_a_late": 2.5e-4,
        "lora_b_middle": 3e-3,
        "lora_b_late": 1e-3,
    }
    grouped_ids = [id(param) for group in groups for param in group["params"]]
    trainable_ids = [id(param) for param in model.parameters() if param.requires_grad]
    assert sorted(grouped_ids) == sorted(trainable_ids)

    optimizer = torch.optim.AdamW(groups)
    update_optimizer_lrs(optimizer, args, epoch=1)
    epoch_one_lrs = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
    assert all(epoch_one_lrs[name] == 1e-3 for name in epoch_one_lrs if name.startswith("lora_"))
    update_optimizer_lrs(optimizer, args, epoch=3)
    epoch_three_lrs = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
    assert epoch_three_lrs == lrs


def test_trainable_ema_only_swaps_trainable_parameters() -> None:
    """EMA selection must leave all frozen backbone weights unchanged."""

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    model[0].weight.requires_grad_(False)
    initial_frozen = model[0].weight.detach().clone()
    ema = TrainableEMA(model, decay=0.5)
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.add_(2.0)
    raw_trainable = model[1].weight.detach().clone()
    ema.update(model)
    backup = ema.apply(model)
    assert torch.equal(model[0].weight, initial_frozen)
    assert torch.allclose(model[1].weight, raw_trainable - 1.0)
    ema.restore(model, backup)
    assert torch.equal(model[1].weight, raw_trainable)


def test_video_flip_preserves_time_and_flips_every_frame_together() -> None:
    """Spatial augmentation must apply consistently across video time."""

    video = torch.arange(2 * 3 * 1 * 2 * 4).reshape(2, 3, 1, 2, 4)
    flipped = apply_video_hflip(video, probability=1.0)
    assert torch.equal(flipped, video.flip(dims=[-1]))
    assert torch.equal(apply_video_hflip(video, probability=0.0), video)


def test_query_dominant_lora_adds_only_one_late_value_rank() -> None:
    """The query-dominant trick must add rank-one value branches late only."""

    backbone = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=0)
    model = FrameBackboneClassifier(backbone, backbone.num_features, temporal="motion_bins", freeze_backbone=True)
    head_params, _ = count_trainable_params(model)
    apply_ssf_to_layernorm(model.backbone, layers="all")
    apply_query_dominant_lora_to_vit(model.backbone)
    trainable, _ = count_trainable_params(model)
    assert trainable - head_params == 46_848
    value_modules = [module for module in model.modules() if getattr(module, "v_a", None) is not None]
    assert len(value_modules) == 4


def test_query_dominant_lora_preserves_baseline_query_initialization() -> None:
    """Paired runs must initialize the shared query branch identically."""

    baseline = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=0)
    candidate = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=0)
    torch.manual_seed(17)
    apply_lora_qv_to_vit(baseline, rank=4, alpha=8, layers="last8", lora_targets="q")
    baseline_query = [module.q_a.weight.detach().clone() for module in baseline.modules() if hasattr(module, "q_a")]
    torch.manual_seed(17)
    apply_query_dominant_lora_to_vit(candidate)
    candidate_query = [module.q_a.weight.detach().clone() for module in candidate.modules() if hasattr(module, "q_a")]
    assert len(baseline_query) == len(candidate_query) == 8
    assert all(torch.equal(left, right) for left, right in zip(baseline_query, candidate_query))
