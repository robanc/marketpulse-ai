import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "monitoring" / "monitoring_snapshot.json"

SOURCES = {
    "Stock Prices": "stock_prices.parquet",
    "FRED Macroeconomic": "fred_macro.parquet",
    "Market Sentiment": "market_sentiment.parquet",
}


def build_snapshot():
    snapshot = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_type": "local_export",
        "data_freshness": [],
        "pipeline": {
            "production_executions": 0,
            "production_successes": 0,
            "production_failures": 0,
            "latest_production_execution": None,
            "test_executions": 0,
        },
    }

    for name, filename in SOURCES.items():
        path = ROOT / "data" / filename
        record = {
            "dataset": name,
            "records": None,
            "latest_observation": None,
            "file_updated_utc": None,
        }

        if path.exists():
            frame = pd.read_parquet(path)
            record["records"] = len(frame)

            if "date" in frame.columns:
                dates = pd.to_datetime(frame["date"], errors="coerce")
                latest = dates.max()
                if pd.notna(latest):
                    record["latest_observation"] = latest.strftime("%Y-%m-%d")

            record["file_updated_utc"] = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat()

        snapshot["data_freshness"].append(record)

    history_path = ROOT / "logs" / "pipeline_execution_history.jsonl"

    if history_path.exists():
        records = []

        for line in history_path.read_text(
            encoding="utf-8-sig"
        ).splitlines():
            if line.strip():
                records.append(json.loads(line))

        production = [
            record for record in records
            if record.get("execution_type") == "production"
        ]
        tests = [
            record for record in records
            if record.get("execution_type") == "test"
        ]

        snapshot["pipeline"]["production_executions"] = len(production)
        snapshot["pipeline"]["production_successes"] = sum(
            record.get("status") == "success" for record in production
        )
        snapshot["pipeline"]["production_failures"] = sum(
            record.get("status") == "failed" for record in production
        )
        snapshot["pipeline"]["test_executions"] = len(tests)

        if production:
            latest = max(
                production,
                key=lambda record: record.get("started_at", ""),
            )
            snapshot["pipeline"]["latest_production_execution"] = {
                "started_at": latest.get("started_at"),
                "completed_at": latest.get("completed_at"),
                "duration_seconds": latest.get("duration_seconds"),
                "status": latest.get("status"),
                "exit_code": latest.get("exit_code"),
            }

    return snapshot


def main():
    snapshot = build_snapshot()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(snapshot, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"PASS: Monitoring snapshot created: {OUTPUT}")
    print(f"Datasets: {len(snapshot['data_freshness'])}")
    print(
        "Production executions:",
        snapshot["pipeline"]["production_executions"],
    )


if __name__ == "__main__":
    main()
