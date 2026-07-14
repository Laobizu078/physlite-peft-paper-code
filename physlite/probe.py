"""Evaluate trained checkpoints under temporal counterfactual transforms."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from .datasets import ManifestVideoDataset
from .experiments import ROOT, absolute
from .models import FrameBackboneClassifier, apply_lora_qv_to_vit, apply_ssf_to_layernorm
from .train import create_timm_model, normalize_video


MODES = ["original", "first_repeat", "last_repeat", "reverse", "endpoint_step", "middle_reverse"]


def transform(video: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply a batch-consistent temporal intervention to ``[B, T, C, H, W]``."""

    if mode == "original":
        return video
    if mode == "first_repeat":
        return video[:, :1].expand_as(video)
    if mode == "last_repeat":
        return video[:, -1:].expand_as(video)
    if mode == "reverse":
        return video.flip(dims=[1])
    if mode == "endpoint_step":
        output = video[:, :1].expand_as(video).clone()
        midpoint = video.shape[1] // 2
        output[:, midpoint:] = video[:, -1:].expand(-1, video.shape[1] - midpoint, -1, -1, -1)
        return output
    if mode == "middle_reverse":
        output = video.clone()
        if video.shape[1] > 2:
            output[:, 1:-1] = video[:, 1:-1].flip(dims=[1])
        return output
    raise ValueError(mode)


def build_model(config: dict, device: torch.device) -> FrameBackboneClassifier:
    """Reconstruct the SSF-LoRA model described by a saved run config."""

    backbone = create_timm_model(config["backbone"], config["pretrained"], config["image_size"])
    model = FrameBackboneClassifier(
        backbone=backbone,
        feature_dim=backbone.num_features,
        temporal=config["temporal"],
        head_hidden=config["head_hidden"],
        head_type=config["head_type"],
        freeze_backbone=True,
    )
    apply_ssf_to_layernorm(model.backbone, layers=config["ssf_layers"])
    apply_lora_qv_to_vit(
        model.backbone,
        rank=config["lora_rank"],
        alpha=config["lora_alpha"],
        dropout=config["lora_dropout"],
        layers=config["lora_layers"],
        lora_targets=config["lora_targets"],
        ia3_targets=config["ia3_targets"],
    )
    return model.to(device)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> dict[str, dict]:
    """Measure all temporal interventions on the same model and samples."""

    values = {mode: {"labels": [], "predictions": [], "true_probabilities": []} for mode in MODES}
    for raw_video, label in loader:
        raw_video = raw_video.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        for mode in MODES:
            video = normalize_video(transform(raw_video, mode), device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(video)
            probability = torch.softmax(logits.float(), dim=-1)
            bucket = values[mode]
            bucket["labels"].extend(label.cpu().tolist())
            bucket["predictions"].extend(logits.argmax(dim=-1).cpu().tolist())
            bucket["true_probabilities"].extend(probability.gather(1, label[:, None]).squeeze(1).cpu().tolist())
    return {
        mode: {
            "acc": float(accuracy_score(bucket["labels"], bucket["predictions"])),
            "balanced_acc": float(balanced_accuracy_score(bucket["labels"], bucket["predictions"])),
            "f1": float(f1_score(bucket["labels"], bucket["predictions"], zero_division=0)),
            "mean_true_probability": float(np.mean(bucket["true_probabilities"])),
        }
        for mode, bucket in values.items()
    }


def parse_args() -> argparse.Namespace:
    """Parse checkpoint, dataset, and probe output locations."""

    parser = argparse.ArgumentParser(description="Run same-model temporal counterfactual probes.")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/counterfactual/rank8_motion_stats")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/main.csv")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/Physion")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/counterfactual/probes.json")
    return parser.parse_args()


def main() -> None:
    """Run counterfactual probes for each seed and aggregate their metrics."""

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the paper probes.")
    device = torch.device("cuda")
    raw = []
    for seed in args.seeds:
        run_path = absolute(args.run_dir) / f"seed_{seed}.json"
        checkpoint_path = absolute(args.run_dir) / f"seed_{seed}.pt"
        if not run_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing run/checkpoint for seed {seed}: {run_path}")
        config = json.loads(run_path.read_text(encoding="utf-8"))
        model = build_model(config, device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        _, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:5]}")
        model.eval()
        dataset = ManifestVideoDataset(
            absolute(args.manifest),
            data_root=absolute(args.data_root),
            split=config.get("test_split", "pilot_test"),
            scenario=config.get("scenario"),
            frames=config["frames"],
            image_size=config["image_size"],
            sampling=config["sampling"],
            observation_seconds=config.get("observation_seconds", 1.5),
        )
        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        raw.append({"seed": seed, "modes": evaluate(model, loader, device, amp=bool(config.get("amp", True)))})
        print(f"counterfactual seed={seed} complete")

    rows = []
    for mode in MODES:
        bacc = [run["modes"][mode]["balanced_acc"] for run in raw]
        f1 = [run["modes"][mode]["f1"] for run in raw]
        probability = [run["modes"][mode]["mean_true_probability"] for run in raw]
        rows.append(
            {
                "mode": mode,
                "n": len(raw),
                "bacc_mean": statistics.mean(bacc),
                "bacc_std": statistics.stdev(bacc) if len(bacc) > 1 else 0.0,
                "f1_mean": statistics.mean(f1),
                "mean_true_probability": statistics.mean(probability),
            }
        )
    original = next(row["bacc_mean"] for row in rows if row["mode"] == "original")
    for row in rows:
        row["drop_from_original"] = original - row["bacc_mean"]
    payload = {"rows": rows, "raw": raw}
    output = absolute(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
