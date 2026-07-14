"""Manifest-backed video datasets and deterministic frame sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class VideoSample:
    """Metadata needed to load one labeled video and audit its split family."""

    video_path: str
    label: int
    video_id: str = "unknown"
    scenario: str = "unknown"
    split: str = "unknown"
    family: str = "unknown"


class SyntheticVideoDataset(Dataset):
    """Tiny synthetic dataset for checking the training loop.

    Labels are intentionally learnable from frame differences: positive samples
    have a brighter final frame than first frame, negatives have the reverse.
    """

    def __init__(
        self,
        size: int = 32,
        frames: int = 4,
        image_size: int = 64,
        seed: int = 0,
    ) -> None:
        self.size = size
        self.frames = frames
        self.image_size = image_size
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        label = idx % 2
        video = self.rng.normal(
            loc=0.45,
            scale=0.12,
            size=(self.frames, 3, self.image_size, self.image_size),
        ).astype("float32")
        ramp = np.linspace(-0.18, 0.18, self.frames, dtype="float32")
        if label == 0:
            ramp = -ramp
        video += ramp[:, None, None, None]
        video = np.clip(video, 0.0, 1.0)
        return torch.from_numpy(video), torch.tensor(label, dtype=torch.long)


class ManifestVideoDataset(Dataset):
    """Loads short videos listed in a CSV manifest.

    Expected columns: video_path,label. Optional: scenario,split.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path | None = None,
        split: str | None = None,
        scenario: str | None = None,
        frames: int = 4,
        image_size: int = 224,
        sampling: Literal["uniform", "early", "last", "observed"] = "uniform",
        observation_seconds: float = 1.5,
    ) -> None:
        df = pd.read_csv(manifest_path)
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split]
        if scenario is not None and "scenario" in df.columns:
            df = df[df["scenario"] == scenario]
        root = Path(data_root).expanduser().resolve() if data_root is not None else None

        def resolve_video_path(raw: str) -> str:
            path = Path(raw).expanduser()
            if path.is_absolute():
                return str(path)
            if root is None:
                raise ValueError("Manifest uses relative video paths; pass data_root.")
            return str((root / path).resolve())

        self.samples = [
            VideoSample(
                video_path=resolve_video_path(str(row.video_path)),
                label=int(row.label),
                video_id=str(getattr(row, "video_id", Path(str(row.video_path)).stem)),
                scenario=str(getattr(row, "scenario", "unknown")),
                split=str(getattr(row, "split", "unknown")),
                family=str(getattr(row, "family", "unknown")),
            )
            for row in df.itertuples(index=False)
        ]
        self.frames = frames
        self.image_size = image_size
        self.sampling = sampling
        self.observation_seconds = observation_seconds

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        video = load_video_frames(
            sample.video_path,
            frames=self.frames,
            image_size=self.image_size,
            sampling=self.sampling,
            observation_seconds=self.observation_seconds,
        )
        return video, torch.tensor(sample.label, dtype=torch.long)


def load_video_frames(
    video_path: str | Path,
    frames: int,
    image_size: int,
    sampling: Literal["uniform", "early", "last", "observed"] = "uniform",
    observation_seconds: float = 1.5,
) -> torch.Tensor:
    """Decode selected RGB frames from a video into a ``[T, C, H, W]`` tensor."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    indices = select_frame_indices(
        total=total,
        fps=fps,
        frames=frames,
        sampling=sampling,
        observation_seconds=observation_seconds,
    )

    out = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frame = frame.astype("float32") / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        out.append(frame)
    cap.release()
    return torch.from_numpy(np.stack(out, axis=0))


def select_frame_indices(
    total: int,
    fps: float,
    frames: int,
    sampling: Literal["uniform", "early", "last", "observed"],
    observation_seconds: float = 1.5,
) -> np.ndarray:
    """Choose frame indices for full-video or observation-prefix protocols."""

    if total <= 0 or frames <= 0:
        raise ValueError("total and frames must be positive.")
    if sampling == "last":
        start = max(0, total - frames)
        indices = np.linspace(start, total - 1, frames).astype(int)
    elif sampling == "observed":
        if fps <= 0:
            raise ValueError("Observed-prefix sampling requires positive FPS.")
        observed_total = min(total, max(frames, int(round(fps * observation_seconds))))
        indices = np.linspace(0, observed_total - 1, frames).astype(int)
    elif sampling == "early":
        end = max(frames, total // 2)
        indices = np.linspace(0, end - 1, frames).astype(int)
    else:
        indices = np.linspace(0, total - 1, frames).astype(int)
    return indices
