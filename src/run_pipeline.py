"""MarketPulse AI pipeline orchestrator."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from pipeline_recovery import create_checkpoint, restore_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PIPELINE_STEPS = [
    "01_data_collection.py",
    "02_eda_feature_engineering.py",
    "03_model_training.py",
    "04_trading_backtest.py",
    "05_strategy_analysis.py",
    "06_momentum_backtest.py",
    "07_latest_predictions.py",
]


INCREMENTAL_STEPS = [
    "08_incremental_stock_prices.py",
    "09_incremental_fred_macro.py",
    "10_incremental_sentiment.py",
    "11_incremental_merge.py",
    "12_incremental_features.py",
    "07_latest_predictions.py",
]


INCREMENTAL_ARGUMENTS = {
    "08_incremental_stock_prices.py": ["--apply"],
    "09_incremental_fred_macro.py": [
        "--hold-unverified",
        "--apply",
    ],
    "10_incremental_sentiment.py": ["--apply"],
    "11_incremental_merge.py": ["--apply"],
    "12_incremental_features.py": ["--apply"],
    "07_latest_predictions.py": [],
}


def main():
    parser = argparse.ArgumentParser(
        description="Run the MarketPulse AI pipeline."
    )

    parser.add_argument(
        "--mode",
        choices=["cached", "full", "incremental"],
        default="cached",
        help="cached uses existing market data; full downloads fresh data.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview pipeline execution without running scripts.",
    )

    args = parser.parse_args()

    if args.mode == "incremental":
        steps = INCREMENTAL_STEPS
    else:
        steps = (
            PIPELINE_STEPS
            if args.mode == "full"
            else PIPELINE_STEPS[1:]
        )

    print(f"MarketPulse AI Pipeline | Mode: {args.mode}")

    # Validate all scripts before creating a checkpoint.
    for step in steps:
        script = PROJECT_ROOT / "src" / "pipeline" / step
        if not script.is_file():
            raise FileNotFoundError(f"Missing script: {script}")

    checkpoint = None

    if args.mode == "incremental" and not args.dry_run:
        checkpoint = create_checkpoint()

    try:
        for step in steps:
            script = PROJECT_ROOT / "src" / "pipeline" / step

            print(
                f"{'[DRY RUN] ' if args.dry_run else '[RUN] '}{step}",
                flush=True,
            )

            if not args.dry_run:
                command = [sys.executable, str(script)]

                if args.mode == "incremental":
                    command.extend(INCREMENTAL_ARGUMENTS[step])

                subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env={**os.environ, "MPLBACKEND": "Agg"},
                    check=True,
                )

    except Exception:
        if checkpoint is not None:
            print(
                "Incremental pipeline failed. Restoring checkpoint...",
                flush=True,
            )
            try:
                restore_checkpoint(checkpoint)
                print("ROLLBACK SUCCESSFUL", flush=True)
            except Exception:
                print(
                    "ROLLBACK FAILED - manual recovery required.",
                    flush=True,
                )
                raise
        raise

    print("Pipeline preview complete." if args.dry_run else "Pipeline completed.")


if __name__ == "__main__":
    main()
