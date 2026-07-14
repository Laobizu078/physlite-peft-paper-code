"""Download Physion-Test-Core and validate the released leakage-free splits."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


DATA_URL = "https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/Physion.zip"
ARCHIVE_SHA256 = "1c80e51d9d299a54cc78bb20b9bb9b597d3b18067fd2f5a06e4e0a3a0c2c0c26"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    """Download a URL atomically through a temporary partial file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def verify_manifests(manifest_dir: Path, data_root: Path) -> None:
    """Validate schema, size, family isolation, and video availability."""

    manifests = [manifest_dir / "main.csv", *sorted((manifest_dir / "repeated_splits").glob("split_*.csv"))]
    if len(manifests) != 6:
        raise RuntimeError(f"Expected main manifest plus five repeated splits, found {len(manifests)} files.")
    for path in manifests:
        frame = pd.read_csv(path)
        required = {"video_path", "video_id", "label", "scenario", "split", "family"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
        if len(frame) != 1200:
            raise RuntimeError(f"{path} has {len(frame)} rows; expected 1200.")
        crossing = int((frame.groupby("family")["split"].nunique() > 1).sum())
        if crossing:
            raise RuntimeError(f"{path} leaks {crossing} families across splits.")
        missing_videos = [raw for raw in frame["video_path"] if not (data_root / raw).is_file()]
        if missing_videos:
            raise RuntimeError(f"{path}: {len(missing_videos)} videos are missing below {data_root}.")
        print(f"OK {path}: rows=1200 families={frame['family'].nunique()} crossing=0")


def parse_args() -> argparse.Namespace:
    """Parse dataset locations and optional offline preparation mode."""

    parser = argparse.ArgumentParser(description="Download Physion-Test-Core and validate the paper manifests.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Prepare the dataset archive and verify every released manifest."""

    args = parse_args()
    archive = args.data_dir / "Physion.zip"
    data_root = args.data_dir / "Physion"
    # Release archives may preserve a local dataset symlink whose target is not
    # available on another machine. Remove only a broken link before extraction.
    if data_root.is_symlink() and not data_root.exists():
        data_root.unlink()
    if not data_root.is_dir():
        if args.skip_download and not archive.is_file():
            raise RuntimeError(f"Dataset not found at {data_root} and --skip-download was used.")
        if not archive.is_file():
            print(f"Downloading {DATA_URL} -> {archive}")
            download(DATA_URL, archive)
        actual = sha256(archive)
        if actual != ARCHIVE_SHA256:
            raise RuntimeError(f"Archive checksum mismatch: {actual}")
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(args.data_dir)
    verify_manifests(args.manifest_dir, data_root)


if __name__ == "__main__":
    main()
