from __future__ import annotations

import argparse
import hashlib
import json
import random
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
    apply_ssf_to_layernorm,
    apply_vpt_to_vit,
    count_trainable_params,
    enable_bitfit,
)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-layers", default="last4")
    parser.add_argument("--lora-targets", default="qv", choices=["q", "v", "qv"])
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
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="outputs/manifest_run.json")
    parser.add_argument("--checkpoint-out", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(requested)


def create_timm_model(model_name: str, pretrained: bool, image_size: int):
    import timm

    try:
        return timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=image_size)
    except TypeError:
        return timm.create_model(model_name, pretrained=pretrained, num_classes=0)


def normalize_video(video: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=device, dtype=video.dtype)
    std = IMAGENET_STD.to(device=device, dtype=video.dtype)
    return (video - mean) / std


def make_temporal_counterfactual(video: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "first_repeat":
        return video[:, :1].expand_as(video)
    if mode == "last_repeat":
        return video[:, -1:].expand_as(video)
    if mode == "reverse":
        return video.flip(dims=[1])
    raise ValueError(f"Unknown temporal contrast mode: {mode}")


def make_loader(args: argparse.Namespace, split: str, shuffle: bool, scenario: str | None = None) -> DataLoader:
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
    )


def evaluate_scenarios(
    args: argparse.Namespace,
    model: torch.nn.Module,
    split: str,
    device: torch.device,
    amp: bool,
) -> dict[str, dict]:
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
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
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
        "ssf_ia3_lora_qv",
        "ssf_phygate_lora",
        "ssf_motion_value_lora",
        "ssf_temporal_lora",
    }:
        motion_value_methods = {"motion_value_lora", "ssf_motion_value_lora"}
        ia3_lora_methods = {"ia3_lora_qv", "ssf_ia3_lora_qv"}
        effective_ia3_targets = "v" if args.method in motion_value_methods else args.ia3_targets
        if args.method in {"ssf_lora_qv", "ssf_ia3_lora_qv", "ssf_phygate_lora", "ssf_motion_value_lora", "ssf_temporal_lora"}:
            peft_modules += apply_ssf_to_layernorm(model.backbone, layers=args.ssf_layers)
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

    head_params = []
    ssf_params = []
    peft_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("head."):
            head_params.append(param)
        elif name.endswith(".ssf_scale") or name.endswith(".ssf_shift"):
            ssf_params.append(param)
        else:
            peft_params.append(param)
    param_groups = []
    if head_params:
        param_groups.append(
            {"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay, "group_name": "head"}
        )
    if ssf_params:
        param_groups.append(
            {
                "params": ssf_params,
                "lr": args.ssf_lr if args.ssf_lr is not None else args.peft_lr,
                "weight_decay": 0.0,
                "group_name": "ssf",
            }
        )
    if peft_params:
        param_groups.append(
            {"params": peft_params, "lr": args.peft_lr, "weight_decay": 0.0, "group_name": "peft"}
        )
    optimizer = torch.optim.AdamW(param_groups)
    loss_fn = torch.nn.CrossEntropyLoss()
    history = []
    best_val = None
    best_state = None
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if args.ssf_warmup_epochs > 0:
            warming_up = epoch <= args.ssf_warmup_epochs
            for group in optimizer.param_groups:
                if group["group_name"] == "peft":
                    group["lr"] = 0.0 if warming_up else args.peft_lr
                elif group["group_name"] == "ssf":
                    ssf_lr = args.ssf_lr if args.ssf_lr is not None else args.peft_lr
                    group["lr"] = 0.0 if (not warming_up and args.freeze_ssf_after_warmup) else ssf_lr
        model.train()
        losses = []
        contrast_losses = []
        correct = 0
        total_seen = 0
        for video, label in train_loader:
            video = normalize_video(video.to(device, non_blocking=True), device)
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
            losses.append(float(loss.item()))
            pred = logits.argmax(dim=-1)
            correct += int((pred == label).sum().item())
            total_seen += int(label.numel())
        val = evaluate(model, val_loader, device, args.amp)
        train_loss = float(np.mean(losses))
        train_acc = correct / max(total_seen, 1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "temporal_contrast_loss": float(np.mean(contrast_losses)) if contrast_losses else 0.0,
            "group_lrs": {group["group_name"]: group["lr"] for group in optimizer.param_groups},
            "val": val,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if best_val is None or val["balanced_acc"] > best_val["balanced_acc"]:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device, args.amp, return_predictions=True)
    test_by_scenario = {} if args.scenario is not None else evaluate_scenarios(args, model, args.test_split, device, args.amp)
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
        "lora_rank": args.lora_rank if args.method in lora_like_methods else None,
        "lora_alpha": args.lora_alpha if args.method in lora_like_methods else None,
        "lora_dropout": args.lora_dropout if args.method in lora_like_methods else None,
        "lora_layers": args.lora_layers if args.method in lora_like_methods else None,
        "lora_targets": args.lora_targets if args.method in lora_like_methods else None,
        "ssf_layers": args.ssf_layers if args.method in {"ssf", "ssf_lora_qv", "ssf_ia3_lora_qv", "ssf_phygate_lora", "ssf_motion_value_lora", "ssf_temporal_lora"} else None,
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
        "seconds": elapsed,
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0,
        "reproducibility": {
            "seed": args.seed,
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
    printable_test = dict(test)
    printable_test["prediction_count"] = len(printable_test.pop("predictions", []))
    printable["test"] = printable_test
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
