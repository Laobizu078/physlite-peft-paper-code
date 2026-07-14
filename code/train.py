"""Train one manifest-defined Physion run with a selected PEFT method."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from .datasets import ManifestVideoDataset
from .models import (
    FrameBackboneClassifier,
    apply_adaptformer_to_vit,
    apply_adapters_to_vit,
    apply_ia3_to_vit,
    apply_lora_qv_to_vit,
    apply_query_dominant_lora_to_vit,
    apply_ssf_to_layernorm,
    apply_vpt_to_vit,
    count_trainable_params,
    enable_bitfit,
)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    """Parse model, PEFT, optimization, protocol, and output options."""

    parser = argparse.ArgumentParser(description="Train one paper configuration on a manifest-based Physion split.")
    parser.add_argument("--manifest", default="data/manifests/main.csv")
    parser.add_argument("--data-root", default="data/Physion", help="Root prepended to relative video paths.")
    parser.add_argument("--backbone", default="deit_small_patch16_224")
    parser.add_argument(
        "--method",
        default="head_only",
        choices=[
            "head_only",
            "bitfit",
            "ssf",
            "adapter",
            "adaptformer",
            "vpt",
            "ia3",
            "lora_qv",
            "ia3_lora_qv",
            "phygate_lora",
            "motion_value_lora",
            "temporal_lora",
            "ssf_lora_qv",
            "ssf_query_dominant_lora",
            "ssf_ia3_lora_qv",
            "ssf_phygate_lora",
            "ssf_motion_value_lora",
            "ssf_temporal_lora",
        ],
    )
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--train-split", default="pilot_train")
    parser.add_argument("--val-split", default="pilot_val")
    parser.add_argument("--test-split", default="pilot_test")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--sampling", default="uniform", choices=["uniform", "early", "last", "observed"])
    parser.add_argument("--observation-seconds", type=float, default=1.5)
    parser.add_argument(
        "--temporal",
        default="motion_diff",
        choices=["mean", "concat", "motion_diff", "motion_stats", "motion_bins", "temporal_attn"],
    )
    parser.add_argument("--head-type", default="mlp", choices=["mlp", "linear", "phys_delta", "phys_residual"])
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument("--phys-delta-hidden", type=int, default=128)
    parser.add_argument("--phys-delta-dropout", type=float, default=0.0)
    parser.add_argument("--phys-residual-scale-init", type=float, default=1.0)
    parser.add_argument("--temporal-contrast-weight", type=float, default=0.0)
    parser.add_argument("--temporal-contrast-margin", type=float, default=0.2)
    parser.add_argument(
        "--temporal-contrast-mode",
        default="first_repeat",
        choices=["first_repeat", "last_repeat", "reverse"],
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--peft-lr", type=float, default=3e-4)
    parser.add_argument(
        "--ssf-lr",
        type=float,
        default=None,
        help="Optional SSF-specific LR for combined methods; defaults to --peft-lr.",
    )
    parser.add_argument(
        "--ssf-warmup-epochs",
        type=int,
        default=0,
        help="Train SSF before enabling the other PEFT parameters in combined methods.",
    )
    parser.add_argument(
        "--freeze-ssf-after-warmup",
        action="store_true",
        help="After SSF warmup, keep SSF fixed while training the remaining PEFT parameters.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--train-hflip-prob",
        type=float,
        default=0.0,
        help="Probability of horizontally mirroring an entire training video; all frames share the transform.",
    )
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-layers", default="last4")
    parser.add_argument("--lora-targets", default="qv", choices=["q", "v", "qv"])
    parser.add_argument("--lora-value-rank", type=int, default=1)
    parser.add_argument("--lora-value-alpha", type=float, default=2.0)
    parser.add_argument("--lora-value-layers", default="last4")
    parser.add_argument(
        "--lora-plus-ratio",
        type=float,
        default=1.0,
        help="LoRA-B/A learning-rate ratio; the geometric mean remains --peft-lr.",
    )
    parser.add_argument(
        "--lora-depth-profile",
        default="uniform",
        choices=["uniform", "middle_focus"],
        help="Optionally emphasize the earlier half of the selected LoRA blocks.",
    )
    parser.add_argument(
        "--lora-depth-strength",
        type=float,
        default=0.5,
        help="Middle-focus strength: earlier/later selected blocks use 1+s and 1-s.",
    )
    parser.add_argument(
        "--lora-allocation-warmup-epochs",
        type=int,
        default=0,
        help="Geometrically introduce LoRA+ and depth allocation over this many epoch transitions.",
    )
    parser.add_argument(
        "--peft-ema-decay",
        type=float,
        default=0.0,
        help="EMA decay for trainable parameters; zero disables trajectory averaging.",
    )
    parser.add_argument(
        "--peft-ema-start-epoch",
        type=int,
        default=1,
        help="First epoch whose optimizer trajectory contributes to the trainable-parameter EMA.",
    )
    parser.add_argument("--ssf-layers", default="all")
    parser.add_argument("--adapter-layers", default="last4")
    parser.add_argument("--adapter-bottleneck", type=int, default=64)
    parser.add_argument("--adaptformer-layers", default="last4")
    parser.add_argument("--adaptformer-bottleneck", type=int, default=64)
    parser.add_argument("--adaptformer-scale", type=float, default=0.1)
    parser.add_argument("--adaptformer-dropout", type=float, default=0.0)
    parser.add_argument("--vpt-prompt-tokens", type=int, default=8)
    parser.add_argument("--vpt-init-std", type=float, default=0.02)
    parser.add_argument("--ia3-layers", default="last4")
    parser.add_argument("--ia3-targets", default="qv")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--loader-seed",
        type=int,
        default=None,
        help="Optional independent training-loader RNG for strictly paired method comparisons.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="outputs/manifest_run.json")
    parser.add_argument("--checkpoint-out", default=None)
    parser.add_argument("--skip-test", action="store_true", help="Skip test and scenario evaluation during validation-only search.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable single-GPU runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    """Resolve a device request and fail explicitly when CUDA is unavailable."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(requested)


def create_timm_model(model_name: str, pretrained: bool, image_size: int):
    """Create a feature-only timm model across fixed and configurable sizes."""

    import timm

    try:
        return timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=image_size)
    except TypeError:
        return timm.create_model(model_name, pretrained=pretrained, num_classes=0)


def normalize_video(video: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Apply ImageNet channel normalization to a batched video tensor."""

    mean = IMAGENET_MEAN.to(device=device, dtype=video.dtype)
    std = IMAGENET_STD.to(device=device, dtype=video.dtype)
    return (video - mean) / std


def apply_video_hflip(video: torch.Tensor, probability: float) -> torch.Tensor:
    """Flip complete videos horizontally without changing temporal order."""

    if probability <= 0.0:
        return video
    mask = torch.rand(video.shape[0], device=video.device) < probability
    if mask.any():
        video = video.clone()
        video[mask] = video[mask].flip(dims=[-1])
    return video


def make_temporal_counterfactual(video: torch.Tensor, mode: str) -> torch.Tensor:
    """Construct a temporal negative used by the optional contrastive loss."""

    if mode == "first_repeat":
        return video[:, :1].expand_as(video)
    if mode == "last_repeat":
        return video[:, -1:].expand_as(video)
    if mode == "reverse":
        return video.flip(dims=[1])
    raise ValueError(f"Unknown temporal contrast mode: {mode}")


def make_loader(args: argparse.Namespace, split: str, shuffle: bool, scenario: str | None = None) -> DataLoader:
    """Create a deterministic manifest loader for one split and scenario."""

    dataset = ManifestVideoDataset(
        args.manifest,
        data_root=args.data_root,
        split=split,
        scenario=args.scenario if scenario is None else scenario,
        frames=args.frames,
        image_size=args.image_size,
        sampling=args.sampling,
        observation_seconds=args.observation_seconds,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No samples for split={split!r} scenario={args.scenario!r}")
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=(
            torch.Generator().manual_seed(args.loader_seed)
            if shuffle and args.loader_seed is not None
            else None
        ),
    )


def evaluate_scenarios(
    args: argparse.Namespace,
    model: torch.nn.Module,
    split: str,
    device: torch.device,
    amp: bool,
) -> dict[str, dict]:
    """Evaluate a model separately on each scenario in a manifest split."""

    df = pd.read_csv(args.manifest)
    if "scenario" not in df.columns:
        return {}
    out = {}
    for scenario in sorted(df["scenario"].dropna().unique().tolist()):
        loader = make_loader(args, split, shuffle=False, scenario=scenario)
        out[scenario] = evaluate(model, loader, device, amp)
    return out


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    return_predictions: bool = False,
) -> dict:
    """Compute classification metrics and optionally serialize predictions."""

    model.eval()
    y_true = []
    y_pred = []
    probabilities = []
    losses = []
    loss_fn = torch.nn.CrossEntropyLoss()
    for video, label in loader:
        video = normalize_video(video.to(device, non_blocking=True), device)
        label = label.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(video)
            loss = loss_fn(logits, label)
        losses.append(float(loss.item()))
        y_true.extend(label.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        if return_predictions:
            probabilities.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist())
    result = {
        "loss": float(np.mean(losses)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    if return_predictions:
        samples = loader.dataset.samples
        if len(samples) != len(y_true):
            raise RuntimeError("Prediction metadata does not match evaluated sample count.")
        result["predictions"] = [
            {
                "video_id": sample.video_id,
                "video_path": sample.video_path,
                "family": sample.family,
                "scenario": sample.scenario,
                "label": int(label),
                "prediction": int(prediction),
                "probability": float(probability),
            }
            for sample, label, prediction, probability in zip(samples, y_true, y_pred, probabilities)
        ]
    return result


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest recorded with each experiment result."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lora_factor(name: str) -> str | None:
    """Classify a trainable parameter as a LoRA A or B factor."""

    if re.search(r"\.(?:q|v)_a\.weight$", name):
        return "a"
    if re.search(r"\.(?:q|v)_b\.weight$", name):
        return "b"
    return None


def _lora_block_index(name: str) -> int | None:
    """Extract a ViT block index from a qualified parameter name."""

    match = re.search(r"\.blocks\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def build_optimizer_groups(model: torch.nn.Module, args: argparse.Namespace) -> list[dict]:
    """Assign head, SSF, and LoRA factors their controlled learning rates."""

    named_trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    selected_blocks = sorted(
        {
            block
            for name, _ in named_trainable
            if _lora_factor(name) is not None
            for block in [_lora_block_index(name)]
            if block is not None
        }
    )
    # The profile redistributes LR within the selected depth at fixed parameters.
    split = (len(selected_blocks) + 1) // 2
    middle_blocks = set(selected_blocks[:split])

    grouped: dict[tuple[str, float, float], list[torch.nn.Parameter]] = {}
    for name, param in named_trainable:
        if name.startswith("head."):
            role, lr, weight_decay = "head", args.lr, args.weight_decay
        elif name.endswith(".ssf_scale") or name.endswith(".ssf_shift"):
            role = "ssf"
            lr = args.ssf_lr if args.ssf_lr is not None else args.peft_lr
            weight_decay = 0.0
        else:
            factor = _lora_factor(name)
            depth_multiplier = 1.0
            depth_label = "uniform"
            if factor is not None and args.lora_depth_profile == "middle_focus":
                block = _lora_block_index(name)
                if block is not None:
                    depth_multiplier = (
                        1.0 + args.lora_depth_strength if block in middle_blocks else 1.0 - args.lora_depth_strength
                    )
                    depth_label = "middle" if block in middle_blocks else "late"
            if factor == "a":
                factor_multiplier = 1.0 / math.sqrt(args.lora_plus_ratio)
                role = f"lora_a_{depth_label}"
            elif factor == "b":
                factor_multiplier = math.sqrt(args.lora_plus_ratio)
                role = f"lora_b_{depth_label}"
            else:
                factor_multiplier = 1.0
                role = "peft"
            lr = args.peft_lr * factor_multiplier * depth_multiplier
            weight_decay = 0.0
        grouped.setdefault((role, lr, weight_decay), []).append(param)

    param_groups = []
    for (name, lr, weight_decay), params in grouped.items():
        group_role = "ssf" if name == "ssf" else "head" if name == "head" else "peft"
        neutral_lr = args.peft_lr if name.startswith("lora_") else lr
        param_groups.append(
            {
                "params": params,
                "lr": lr,
                "base_lr": lr,
                "neutral_lr": neutral_lr,
                "weight_decay": weight_decay,
                "group_name": name,
                "group_role": group_role,
            }
        )
    return param_groups


def update_optimizer_lrs(optimizer: torch.optim.Optimizer, args: argparse.Namespace, epoch: int) -> None:
    """Apply allocation warmup and optional staged SSF/LoRA training."""

    allocation_warmup = args.lora_allocation_warmup_epochs
    allocation_progress = 1.0 if allocation_warmup == 0 else min((epoch - 1) / allocation_warmup, 1.0)
    for group in optimizer.param_groups:
        target_lr = group["base_lr"]
        neutral_lr = group["neutral_lr"]
        if group["group_name"].startswith("lora_"):
            target_lr = neutral_lr * (target_lr / neutral_lr) ** allocation_progress
        group["lr"] = target_lr

    if args.ssf_warmup_epochs > 0:
        warming_up = epoch <= args.ssf_warmup_epochs
        for group in optimizer.param_groups:
            if group["group_role"] == "peft":
                group["lr"] = 0.0 if warming_up else group["lr"]
            elif group["group_role"] == "ssf":
                ssf_lr = args.ssf_lr if args.ssf_lr is not None else args.peft_lr
                group["lr"] = 0.0 if (not warming_up and args.freeze_ssf_after_warmup) else ssf_lr


class TrainableEMA:
    """Track and temporarily apply EMA weights for trainable parameters only."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Update shadow weights from the current trainable parameters."""

        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(param.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def apply(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Swap EMA weights into the model and return the original weights."""

        backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.detach().clone()
                param.copy_(self.shadow[name])
        return backup

    @staticmethod
    @torch.no_grad()
    def restore(model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        """Restore parameters returned by :meth:`apply`."""

        for name, param in model.named_parameters():
            if name in backup:
                param.copy_(backup[name])


def main() -> None:
    """Train, select by validation BAcc, evaluate, and serialize one run."""

    args = parse_args()
    if args.lora_plus_ratio < 1.0:
        raise ValueError("--lora-plus-ratio must be at least 1.")
    if not 0.0 <= args.lora_depth_strength < 1.0:
        raise ValueError("--lora-depth-strength must be in [0, 1).")
    if args.lora_allocation_warmup_epochs < 0 or args.lora_allocation_warmup_epochs >= args.epochs:
        raise ValueError("--lora-allocation-warmup-epochs must be in [0, epochs).")
    if args.peft_ema_decay < 0.0 or args.peft_ema_decay >= 1.0:
        raise ValueError("--peft-ema-decay must be in [0, 1).")
    if args.peft_ema_start_epoch < 1 or args.peft_ema_start_epoch > args.epochs:
        raise ValueError("--peft-ema-start-epoch must be in [1, epochs].")
    if not 0.0 <= args.train_hflip_prob <= 1.0:
        raise ValueError("--train-hflip-prob must be in [0, 1].")
    if args.ssf_warmup_epochs < 0 or args.ssf_warmup_epochs >= args.epochs:
        raise ValueError("--ssf-warmup-epochs must be in [0, epochs).")
    if args.ssf_warmup_epochs > 0 and not args.method.startswith("ssf_"):
        raise ValueError("SSF warmup requires a combined method whose name starts with 'ssf_'.")
    if args.freeze_ssf_after_warmup and args.ssf_warmup_epochs == 0:
        raise ValueError("--freeze-ssf-after-warmup requires --ssf-warmup-epochs > 0.")
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device={device}")
    if device.type == "cuda":
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    train_loader = make_loader(args, args.train_split, shuffle=True)
    val_loader = make_loader(args, args.val_split, shuffle=False)
    test_loader = make_loader(args, args.test_split, shuffle=False)

    backbone = create_timm_model(args.backbone, args.pretrained, args.image_size)
    model = FrameBackboneClassifier(
        backbone=backbone,
        feature_dim=backbone.num_features,
        temporal=args.temporal,
        head_hidden=args.head_hidden,
        head_type=args.head_type,
        phys_delta_hidden=args.phys_delta_hidden,
        phys_delta_dropout=args.phys_delta_dropout,
        phys_residual_scale_init=args.phys_residual_scale_init,
        freeze_backbone=True,
    ).to(device)
    peft_modules = 0
    if args.method == "bitfit":
        peft_modules = enable_bitfit(model.backbone)
    elif args.method == "ssf":
        peft_modules = apply_ssf_to_layernorm(model.backbone, layers=args.ssf_layers)
        model.to(device)
    elif args.method == "adapter":
        peft_modules = apply_adapters_to_vit(
            model.backbone,
            bottleneck=args.adapter_bottleneck,
            layers=args.adapter_layers,
        )
        model.to(device)
    elif args.method == "adaptformer":
        peft_modules = apply_adaptformer_to_vit(
            model.backbone,
            bottleneck=args.adaptformer_bottleneck,
            layers=args.adaptformer_layers,
            scale=args.adaptformer_scale,
            dropout=args.adaptformer_dropout,
        )
        model.to(device)
    elif args.method == "vpt":
        model.backbone = apply_vpt_to_vit(
            model.backbone,
            prompt_tokens=args.vpt_prompt_tokens,
            init_std=args.vpt_init_std,
        )
        peft_modules = 1
        model.to(device)
    elif args.method == "ia3":
        peft_modules = apply_ia3_to_vit(
            model.backbone,
            layers=args.ia3_layers,
            targets=args.ia3_targets,
        )
        model.to(device)
    elif args.method in {
        "lora_qv",
        "ia3_lora_qv",
        "phygate_lora",
        "motion_value_lora",
        "temporal_lora",
        "ssf_lora_qv",
        "ssf_query_dominant_lora",
        "ssf_ia3_lora_qv",
        "ssf_phygate_lora",
        "ssf_motion_value_lora",
        "ssf_temporal_lora",
    }:
        motion_value_methods = {"motion_value_lora", "ssf_motion_value_lora"}
        ia3_lora_methods = {"ia3_lora_qv", "ssf_ia3_lora_qv"}
        effective_ia3_targets = "v" if args.method in motion_value_methods else args.ia3_targets
        if args.method in {"ssf_lora_qv", "ssf_query_dominant_lora", "ssf_ia3_lora_qv", "ssf_phygate_lora", "ssf_motion_value_lora", "ssf_temporal_lora"}:
            peft_modules += apply_ssf_to_layernorm(model.backbone, layers=args.ssf_layers)
        if args.method == "ssf_query_dominant_lora":
            peft_modules += apply_query_dominant_lora_to_vit(
                model.backbone,
                query_rank=args.lora_rank,
                query_alpha=args.lora_alpha,
                query_layers=args.lora_layers,
                value_rank=args.lora_value_rank,
                value_alpha=args.lora_value_alpha,
                value_layers=args.lora_value_layers,
                dropout=args.lora_dropout,
            )
        else:
            peft_modules += apply_lora_qv_to_vit(
                model.backbone,
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                layers=args.lora_layers,
                lora_targets=args.lora_targets,
                phygate=args.method in {"phygate_lora", "ssf_phygate_lora"},
                ia3_targets=effective_ia3_targets if args.method in ia3_lora_methods else None,
                motion_value_gate=args.method in motion_value_methods,
                temporal_conditioning=args.method in {"temporal_lora", "ssf_temporal_lora"},
            )
        model.to(device)
    trainable, total = count_trainable_params(model)
    print(f"params trainable={trainable:,} total={total:,} ratio={trainable / total:.4%}")
    if peft_modules:
        print(f"peft method={args.method} modules={peft_modules}")
    lora_like_methods = {
        "lora_qv",
        "ia3_lora_qv",
        "phygate_lora",
        "motion_value_lora",
        "temporal_lora",
        "ssf_lora_qv",
        "ssf_query_dominant_lora",
        "ssf_ia3_lora_qv",
        "ssf_phygate_lora",
        "ssf_motion_value_lora",
        "ssf_temporal_lora",
    }
    if args.method in lora_like_methods:
        print(
            f"lora rank={args.lora_rank} alpha={args.lora_alpha} "
            f"dropout={args.lora_dropout} layers={args.lora_layers} targets={args.lora_targets}"
        )
        if args.method in {"ia3_lora_qv", "ssf_ia3_lora_qv", "motion_value_lora", "ssf_motion_value_lora"}:
            print(f"ia3 targets={'v' if args.method in {'motion_value_lora', 'ssf_motion_value_lora'} else args.ia3_targets}")
    print(f"samples train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    param_groups = build_optimizer_groups(model, args)
    optimizer = torch.optim.AdamW(param_groups)
    loss_fn = torch.nn.CrossEntropyLoss()
    history = []
    best_val = None
    best_state = None
    best_variant = None
    ema = None
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        update_optimizer_lrs(optimizer, args, epoch)
        if args.peft_ema_decay > 0.0 and epoch == args.peft_ema_start_epoch:
            ema = TrainableEMA(model, args.peft_ema_decay)
        model.train()
        losses = []
        contrast_losses = []
        correct = 0
        total_seen = 0
        for video, label in train_loader:
            video = video.to(device, non_blocking=True)
            video = apply_video_hflip(video, args.train_hflip_prob)
            video = normalize_video(video, device)
            label = label.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
                logits = model(video)
                loss = loss_fn(logits, label)
                if args.temporal_contrast_weight > 0:
                    counterfactual = make_temporal_counterfactual(video, args.temporal_contrast_mode)
                    cf_logits = model(counterfactual)
                    true_logits = logits.gather(1, label[:, None]).squeeze(1)
                    cf_true_logits = cf_logits.gather(1, label[:, None]).squeeze(1)
                    contrast_loss = torch.relu(
                        args.temporal_contrast_margin - (true_logits - cf_true_logits)
                    ).mean()
                    loss = loss + args.temporal_contrast_weight * contrast_loss
                    contrast_losses.append(float(contrast_loss.item()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)
            losses.append(float(loss.item()))
            pred = logits.argmax(dim=-1)
            correct += int((pred == label).sum().item())
            total_seen += int(label.numel())
        raw_val = evaluate(model, val_loader, device, args.amp)
        ema_val = None
        selected_variant = "raw"
        selected_val = raw_val
        selected_state = None
        if ema is not None:
            cpu_rng_state = torch.random.get_rng_state()
            cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            backup = ema.apply(model)
            ema_val = evaluate(model, val_loader, device, args.amp)
            if ema_val["balanced_acc"] > raw_val["balanced_acc"]:
                selected_variant = "ema"
                selected_val = ema_val
                selected_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            ema.restore(model, backup)
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
        train_loss = float(np.mean(losses))
        train_acc = correct / max(total_seen, 1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "temporal_contrast_loss": float(np.mean(contrast_losses)) if contrast_losses else 0.0,
            "group_lrs": {group["group_name"]: group["lr"] for group in optimizer.param_groups},
            "val": selected_val,
            "raw_val": raw_val,
            "ema_val": ema_val,
            "selected_variant": selected_variant,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if best_val is None or selected_val["balanced_acc"] > best_val["balanced_acc"]:
            best_val = selected_val
            best_variant = selected_variant
            best_state = selected_state or {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test = None if args.skip_test else evaluate(model, test_loader, device, args.amp, return_predictions=True)
    test_by_scenario = (
        {}
        if args.skip_test or args.scenario is not None
        else evaluate_scenarios(args, model, args.test_split, device, args.amp)
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = {
        "backbone": args.backbone,
        "pretrained": args.pretrained,
        "scenario": args.scenario,
        "method": args.method,
        "batch_size": args.batch_size,
        "frames": args.frames,
        "image_size": args.image_size,
        "sampling": args.sampling,
        "observation_seconds": args.observation_seconds,
        "temporal": args.temporal,
        "head_type": args.head_type,
        "head_hidden": args.head_hidden,
        "phys_delta_hidden": args.phys_delta_hidden if args.head_type == "phys_residual" else None,
        "phys_delta_dropout": args.phys_delta_dropout if args.head_type == "phys_residual" else None,
        "phys_residual_scale_init": args.phys_residual_scale_init if args.head_type == "phys_residual" else None,
        "temporal_contrast_weight": args.temporal_contrast_weight,
        "temporal_contrast_margin": args.temporal_contrast_margin,
        "temporal_contrast_mode": args.temporal_contrast_mode,
        "epochs": args.epochs,
        "lr": args.lr,
        "peft_lr": args.peft_lr,
        "ssf_lr": args.ssf_lr if args.ssf_lr is not None else args.peft_lr,
        "ssf_warmup_epochs": args.ssf_warmup_epochs,
        "freeze_ssf_after_warmup": args.freeze_ssf_after_warmup,
        "weight_decay": args.weight_decay,
        "train_hflip_prob": args.train_hflip_prob,
        "lora_rank": args.lora_rank if args.method in lora_like_methods else None,
        "lora_alpha": args.lora_alpha if args.method in lora_like_methods else None,
        "lora_dropout": args.lora_dropout if args.method in lora_like_methods else None,
        "lora_layers": args.lora_layers if args.method in lora_like_methods else None,
        "lora_targets": args.lora_targets if args.method in lora_like_methods else None,
        "lora_value_rank": args.lora_value_rank if args.method == "ssf_query_dominant_lora" else None,
        "lora_value_alpha": args.lora_value_alpha if args.method == "ssf_query_dominant_lora" else None,
        "lora_value_layers": args.lora_value_layers if args.method == "ssf_query_dominant_lora" else None,
        "lora_plus_ratio": args.lora_plus_ratio if args.method in lora_like_methods else None,
        "lora_depth_profile": args.lora_depth_profile if args.method in lora_like_methods else None,
        "lora_depth_strength": args.lora_depth_strength if args.method in lora_like_methods else None,
        "lora_allocation_warmup_epochs": args.lora_allocation_warmup_epochs if args.method in lora_like_methods else None,
        "peft_ema_decay": args.peft_ema_decay,
        "peft_ema_start_epoch": args.peft_ema_start_epoch if args.peft_ema_decay > 0.0 else None,
        "best_variant": best_variant,
        "ssf_layers": args.ssf_layers if args.method in {"ssf", "ssf_lora_qv", "ssf_query_dominant_lora", "ssf_ia3_lora_qv", "ssf_phygate_lora", "ssf_motion_value_lora", "ssf_temporal_lora"} else None,
        "adapter_layers": args.adapter_layers if args.method == "adapter" else None,
        "adapter_bottleneck": args.adapter_bottleneck if args.method == "adapter" else None,
        "adaptformer_layers": args.adaptformer_layers if args.method == "adaptformer" else None,
        "adaptformer_bottleneck": args.adaptformer_bottleneck if args.method == "adaptformer" else None,
        "adaptformer_scale": args.adaptformer_scale if args.method == "adaptformer" else None,
        "adaptformer_dropout": args.adaptformer_dropout if args.method == "adaptformer" else None,
        "vpt_prompt_tokens": args.vpt_prompt_tokens if args.method == "vpt" else None,
        "vpt_init_std": args.vpt_init_std if args.method == "vpt" else None,
        "ia3_layers": args.ia3_layers if args.method == "ia3" else None,
        "ia3_targets": (
            "v"
            if args.method in {"motion_value_lora", "ssf_motion_value_lora"}
            else args.ia3_targets
            if args.method in {"ia3", "ia3_lora_qv", "ssf_ia3_lora_qv"}
            else None
        ),
        "peft_modules": peft_modules,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": trainable / total,
        "best_val": best_val,
        "test": test,
        "test_by_scenario": test_by_scenario,
        "history": history,
        "skip_test": args.skip_test,
        "seconds": elapsed,
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0,
        "reproducibility": {
            "seed": args.seed,
            "loader_seed": args.loader_seed,
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": file_sha256(args.manifest),
            "data_root": str(Path(args.data_root).resolve()),
            "train_split": args.train_split,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "command": shlex.join(sys.argv),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device),
        },
    }
    if args.checkpoint_out:
        trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
        trainable_state = {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
            if name in trainable_names
        }
        checkpoint = {
            "state_dict": trainable_state,
            "trainable_only": True,
            "config": result,
        }
        ckpt_path = Path(args.checkpoint_out)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, ckpt_path)
        result["checkpoint_out"] = str(ckpt_path)
    printable = dict(result)
    if test is not None:
        printable_test = dict(test)
        printable_test["prediction_count"] = len(printable_test.pop("predictions", []))
        printable["test"] = printable_test
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
