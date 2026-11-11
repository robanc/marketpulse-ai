"""MarketPulse AI pipeline orchestrator."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(
        description="Run the MarketPulse AI pipeline."
    )

    parser.add_argument(
        "--mode",
        choices=["cached", "full"],
        default="cached",
        help="cached uses existing market data; full downloads fresh data.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview pipeline execution without running scripts.",
    )

    args = parser.parse_args()

    steps = (
        PIPELINE_STEPS
        if args.mode == "full"
        else PIPELINE_STEPS[1:]
    )

    print(f"MarketPulse AI Pipeline | Mode: {args.mode}")

    for step in steps:
        script = PROJECT_ROOT / "src" / "pipeline" / step

        if not script.is_file():
            raise FileNotFoundError(f"Missing script: {script}")

        print(f"{'[DRY RUN] ' if args.dry_run else '[RUN] '}{step}", flush=True)

        if not args.dry_run:
            subprocess.run(
                [sys.executable, str(script)],
                cwd=PROJECT_ROOT,
                env={**os.environ, 'MPLBACKEND': 'Agg'},
                check=True,
            )

    print("Pipeline preview complete." if args.dry_run else "Pipeline completed.")


if __name__ == "__main__":
    main()
