"""Validated, non-destructive incremental Parquet merging."""

from pathlib import Path
import pandas as pd


def merge_incremental(existing, incoming, keys):
    """Merge records, with incoming values taking precedence on matching keys."""
    if not isinstance(existing, pd.DataFrame) or not isinstance(incoming, pd.DataFrame):
        raise TypeError("Both inputs must be pandas DataFrames.")

    keys = list(keys)
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Keys must be a nonempty list of unique column names.")

    if set(existing.columns) != set(incoming.columns):
        raise ValueError(
            "Schema mismatch: "
            f"existing={list(existing.columns)}, incoming={list(incoming.columns)}"
        )

    old = existing.copy()
    new = incoming[old.columns].copy()

    for frame in (old, new):
        if frame[keys].isna().any().any():
            raise ValueError("Null values found in primary keys.")
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate primary keys found within an input dataset.")

    if "date" in keys:
        old["date"] = pd.to_datetime(old["date"]).astype("datetime64[ns]")
        new["date"] = pd.to_datetime(new["date"]).astype("datetime64[ns]")

    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=keys, keep="last")
    combined = combined.sort_values(keys).reset_index(drop=True)

    if combined.duplicated(keys).any():
        raise AssertionError("Duplicate primary keys after merge.")

    return combined


def preview_incremental(existing_path, incoming_path, keys):
    """Preview the merge without writing any files."""
    existing_path = Path(existing_path)
    incoming_path = Path(incoming_path)

    old = pd.read_parquet(existing_path)
    new = pd.read_parquet(incoming_path)
    merged = merge_incremental(old, new, keys)

    return {
        "existing_rows": len(old),
        "incoming_rows": len(new),
        "merged_rows": len(merged),
        "net_row_change": len(merged) - len(old),
        "duplicates_after_merge": int(merged.duplicated(keys).sum()),
    }
