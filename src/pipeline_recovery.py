"""Verified checkpoints for MarketPulse AI incremental runs."""

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

PROTECTED_FILES = [
    "data/stock_prices.parquet",
    "data/fred_macro.parquet",
    "data/market_sentiment.parquet",
    "data/marketpulse_merged.parquet",
    "data/marketpulse_features.parquet",
    "data/marketpulse_training.parquet",
    "results/marketpulse_latest_predictions.csv",
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_checkpoint():
    checkpoint_root = ROOT / "data" / "pipeline_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    checkpoint = checkpoint_root / (
        "checkpoint_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    checkpoint.mkdir(exist_ok=False)

    manifest = {}

    for relative in PROTECTED_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)

        destination = checkpoint / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

        original_hash = sha256(source)
        backup_hash = sha256(destination)

        if original_hash != backup_hash:
            raise RuntimeError(f"Backup verification failed: {relative}")

        manifest[relative] = {
            "sha256": original_hash,
            "bytes": source.stat().st_size,
        }

    (checkpoint / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("Recovery checkpoint:", checkpoint, flush=True)
    print("Protected files:", len(manifest), flush=True)

    return checkpoint


def restore_checkpoint(checkpoint):
    checkpoint = Path(checkpoint)
    manifest_path = checkpoint / "manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if set(manifest) != set(PROTECTED_FILES):
        raise ValueError("Checkpoint manifest file list mismatch")

    # Verify every backup before restoring any file.
    for relative, metadata in manifest.items():
        backup = checkpoint / relative
        if not backup.is_file():
            raise FileNotFoundError(backup)

        if sha256(backup) != metadata["sha256"]:
            raise RuntimeError(f"Corrupt checkpoint: {relative}")

    # Restore each file through a temporary file on the same filesystem.
    for relative, metadata in manifest.items():
        destination = ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)

        fd, temporary = tempfile.mkstemp(
            prefix=destination.stem + "_restore_",
            dir=destination.parent,
        )
        os.close(fd)

        try:
            shutil.copyfile(checkpoint / relative, temporary)

            if sha256(temporary) != metadata["sha256"]:
                raise RuntimeError(
                    f"Restore verification failed: {relative}"
                )

            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    print("Checkpoint restored:", checkpoint, flush=True)
    print("Restored files:", len(manifest), flush=True)
