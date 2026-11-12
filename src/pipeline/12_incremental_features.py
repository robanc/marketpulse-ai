"""MarketPulse AI feature engineering preview.

Reuses calculations from Notebook 02.
Does not modify existing datasets.
"""

from pathlib import Path
import argparse
import os
import tempfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

df = pd.read_parquet(DATA / "marketpulse_merged.parquet")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

print("MARKETPULSE AI - FEATURE ENGINEERING PREVIEW")
print("=" * 75)

features_df = df.copy()

features_df = (
    features_df
    .sort_values(["ticker", "date"])
    .reset_index(drop=True)
)

grouped = features_df.groupby("ticker")["close"]

features_df["return_1d"] = grouped.pct_change(1)

features_df["return_5d"] = grouped.pct_change(5)

features_df["return_10d"] = grouped.pct_change(10)

features_df["return_20d"] = grouped.pct_change(20)

for window in [5, 10, 20, 50]:
    features_df[f"sma_{window}"] = (
        features_df.groupby("ticker")["close"]
        .transform(
            lambda x: x.rolling(
                window=window,
                min_periods=window
            ).mean()
        )
    )

for span in [12, 26]:
    features_df[f"ema_{span}"] = (
        features_df.groupby("ticker")["close"]
        .transform(
            lambda x: x.ewm(
                span=span,
                adjust=False,
                min_periods=span
            ).mean()
        )
    )

for window in [5, 20]:
    features_df[f"volatility_{window}d"] = (
        features_df.groupby("ticker")["return_1d"]
        .transform(
            lambda x: x.rolling(
                window=window,
                min_periods=window
            ).std() * np.sqrt(252)
        )
    )

features_df["daily_range_pct"] = (
    (features_df["high"] - features_df["low"]) /
    features_df["close"]
)

previous_close = (
    features_df.groupby("ticker")["close"].shift(1)
)

true_range = pd.concat(
    [
        features_df["high"] - features_df["low"],
        (features_df["high"] - previous_close).abs(),
        (features_df["low"] - previous_close).abs()
    ],
    axis=1
).max(axis=1)

features_df["atr_14"] = (
    true_range.groupby(features_df["ticker"])
    .transform(
        lambda x: x.rolling(
            window=14,
            min_periods=14
        ).mean()
    )
)

price_change = features_df.groupby("ticker")["close"].diff()

gain = price_change.clip(lower=0)

loss = -price_change.clip(upper=0)

avg_gain = gain.groupby(features_df["ticker"]).transform(
    lambda x: x.rolling(14, min_periods=14).mean()
)

avg_loss = loss.groupby(features_df["ticker"]).transform(
    lambda x: x.rolling(14, min_periods=14).mean()
)

rs = avg_gain / avg_loss.replace(0, np.nan)

features_df["rsi_14"] = 100 - (100 / (1 + rs))

features_df.loc[
    (avg_loss == 0) & (avg_gain > 0), "rsi_14"
] = 100

features_df.loc[
    (avg_loss == 0) & (avg_gain == 0), "rsi_14"
] = 50

features_df["macd"] = (
    features_df["ema_12"] - features_df["ema_26"]
)

features_df["macd_signal"] = (
    features_df.groupby("ticker")["macd"]
    .transform(
        lambda x: x.ewm(
            span=9,
            adjust=False,
            min_periods=9
        ).mean()
    )
)

features_df["macd_histogram"] = (
    features_df["macd"] -
    features_df["macd_signal"]
)

rolling_std_20 = (
    features_df.groupby("ticker")["close"]
    .transform(
        lambda x: x.rolling(
            window=20,
            min_periods=20
        ).std()
    )
)

features_df["bollinger_upper"] = (
    features_df["sma_20"] + 2 * rolling_std_20
)

features_df["bollinger_lower"] = (
    features_df["sma_20"] - 2 * rolling_std_20
)

features_df["volume_change_1d"] = (
    features_df.groupby("ticker")["volume"]
    .pct_change()
)

features_df["volume_sma_20"] = (
    features_df.groupby("ticker")["volume"]
    .transform(
        lambda x: x.rolling(
            window=20,
            min_periods=20
        ).mean()
    )
)

features_df["relative_volume_20"] = (
    features_df["volume"] /
    features_df["volume_sma_20"]
)

features_df["log_volume"] = np.log1p(
    features_df["volume"]
)

macro_daily = (
    features_df[
        [
            "date",
            "treasury_10y",
            "treasury_2y",
            "fed_funds_rate",
            "cpi"
        ]
    ]
    .drop_duplicates(subset="date")
    .sort_values("date")
    .reset_index(drop=True)
)

macro_daily["yield_spread"] = (
    macro_daily["treasury_10y"] -
    macro_daily["treasury_2y"]
)

macro_daily["treasury_10y_change_5d"] = (
    macro_daily["treasury_10y"].diff(5)
)

macro_daily["treasury_2y_change_5d"] = (
    macro_daily["treasury_2y"].diff(5)
)

macro_daily["fed_funds_change_20d"] = (
    macro_daily["fed_funds_rate"].diff(20)
)

macro_daily["cpi_change_60d"] = (
    macro_daily["cpi"].pct_change(60)
)

macro_feature_columns = [
    "yield_spread",
    "treasury_10y_change_5d",
    "treasury_2y_change_5d",
    "fed_funds_change_20d",
    "cpi_change_60d"
]

features_df = features_df.merge(
    macro_daily[["date"] + macro_feature_columns],
    on="date",
    how="left",
    validate="many_to_one"
)

sentiment_daily = (
    features_df[
        ["date", "fear_greed_index"]
    ]
    .drop_duplicates(subset="date")
    .sort_values("date")
    .reset_index(drop=True)
)

sentiment_daily["sentiment_change_1d"] = (
    sentiment_daily["fear_greed_index"].diff(1)
)

sentiment_daily["sentiment_sma_5"] = (
    sentiment_daily["fear_greed_index"]
    .rolling(window=5, min_periods=5)
    .mean()
)

sentiment_daily["sentiment_sma_20"] = (
    sentiment_daily["fear_greed_index"]
    .rolling(window=20, min_periods=20)
    .mean()
)

sentiment_daily["sentiment_volatility_20"] = (
    sentiment_daily["fear_greed_index"]
    .rolling(window=20, min_periods=20)
    .std()
)

sentiment_feature_columns = [
    "sentiment_change_1d",
    "sentiment_sma_5",
    "sentiment_sma_20",
    "sentiment_volatility_20"
]

features_df = features_df.merge(
    sentiment_daily[
        ["date"] + sentiment_feature_columns
    ],
    on="date",
    how="left",
    validate="many_to_one"
)

features_df = (
    features_df
    .sort_values(["ticker", "date"])
    .reset_index(drop=True)
)

features_df["future_close_10d"] = (
    features_df.groupby("ticker")["close"]
    .shift(-10)
)

features_df["target_return_10d"] = (
    features_df["future_close_10d"] /
    features_df["close"] - 1
)

features_df["target_direction_10d"] = (
    features_df["target_return_10d"]
    .gt(0)
    .astype("Int64")
    .where(features_df["target_return_10d"].notna())
)

feature_columns = [
    # Price momentum
    "return_1d", "return_5d",
    "return_10d", "return_20d",

    # Moving averages
    "sma_5", "sma_10", "sma_20", "sma_50",
    "ema_12", "ema_26",

    # Volatility
    "volatility_5d", "volatility_20d",
    "daily_range_pct", "atr_14",

    # Technical indicators
    "rsi_14", "macd", "macd_signal",
    "macd_histogram",
    "bollinger_upper", "bollinger_lower",

    # Volume
    "volume_change_1d", "volume_sma_20",
    "relative_volume_20", "log_volume",

    # Macroeconomic
    "yield_spread",
    "treasury_10y_change_5d",
    "treasury_2y_change_5d",
    "fed_funds_change_20d",
    "cpi_change_60d",

    # Sentiment
    "sentiment_change_1d",
    "sentiment_sma_5",
    "sentiment_sma_20",
    "sentiment_volatility_20"
]

target_columns = [
    "target_return_10d",
    "target_direction_10d"
]

complete_feature_mask = (
    features_df[feature_columns].notna().all(axis=1)
)

model_ready_mask = (
    complete_feature_mask
    & features_df[target_columns].notna().all(axis=1)
)

training_df = (
    features_df.loc[
        model_ready_mask & (features_df["ticker"] != "SPY")
    ]
    .copy()
    .sort_values(["date", "ticker"])
    .reset_index(drop=True)
    .drop(columns=["future_close_10d"])
)

print("Merged input:", len(df))
print("Engineered records:", len(features_df))
print("Feature columns:", len(feature_columns))
print("Training records:", len(training_df))
print("Latest feature date:", features_df["date"].max().date())
print("Latest training date:", training_df["date"].max().date())


parser = argparse.ArgumentParser()
parser.add_argument(
    "--apply",
    action="store_true",
    help="Save validated feature and training datasets",
)
args = parser.parse_args()

feature_path = DATA / "marketpulse_features.parquet"
training_path = DATA / "marketpulse_training.parquet"

old_features = pd.read_parquet(feature_path)
old_training = pd.read_parquet(training_path)

for frame in (old_features, old_training, features_df, training_df):
    frame["date"] = pd.to_datetime(
        frame["date"]
    ).astype("datetime64[ns]")

keys = ["ticker", "date"]

old_indexed = old_features.set_index(keys).sort_index()
new_indexed = features_df.set_index(keys).sort_index()

assert old_indexed.index.isin(new_indexed.index).all()
assert len(feature_columns) == 33
assert len(features_df) == len(df)
assert not features_df.duplicated(keys).any()
assert not training_df.duplicated(keys).any()
assert not training_df[feature_columns + target_columns].isna().any().any()
assert "SPY" not in set(training_df["ticker"])

historical = new_indexed.loc[old_indexed.index]

for column in feature_columns:
    assert np.isclose(
        old_indexed[column].to_numpy(dtype=float),
        historical[column].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    ).all(), f"Historical mismatch: {column}"

known = old_indexed["target_return_10d"].notna()

assert np.isclose(
    old_indexed.loc[known, "target_return_10d"].to_numpy(dtype=float),
    historical.loc[known, "target_return_10d"].to_numpy(dtype=float),
    rtol=1e-10,
    atol=1e-10,
).all(), "Previously known targets changed"

old_training_keys = old_training.set_index(keys).index
new_training_keys = training_df.set_index(keys).index

assert old_training_keys.isin(new_training_keys).all()

print("\nVALIDATION: PASSED")
print("Historical predictive features: MATCH")
print("Previously known targets: MATCH")
print("Previous training rows: PRESERVED")
print("New training rows:", len(training_df) - len(old_training))

if not args.apply:
    print("\nPREVIEW ONLY - No datasets modified")
else:
    from datetime import datetime
    import shutil

    backup_dir = (
        DATA / "backups" /
        f"pre_feature_update_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    backup_dir.mkdir(parents=True, exist_ok=False)

    for original in (feature_path, training_path):
        backup_file = backup_dir / original.name
        shutil.copy2(original, backup_file)

        import hashlib
        original_hash = hashlib.sha256(
            original.read_bytes()
        ).hexdigest()
        backup_hash = hashlib.sha256(
            backup_file.read_bytes()
        ).hexdigest()

        assert original_hash == backup_hash, (
            f"Backup verification failed: {original.name}"
        )

    print("\nVerified backup:", backup_dir)

    def atomic_save(frame, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=destination.stem + "_",
            suffix=".parquet",
            dir=destination.parent,
        )
        os.close(fd)

        try:
            frame.to_parquet(temporary, index=False)
            verified = pd.read_parquet(temporary)
            assert verified.shape == frame.shape
            assert list(verified.columns) == list(frame.columns)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    atomic_save(
        features_df,
        DATA / "marketpulse_features.parquet",
    )
    atomic_save(
        training_df,
        DATA / "marketpulse_training.parquet",
    )

    print("\nUPDATED: marketpulse_features.parquet")
    print("UPDATED: marketpulse_training.parquet")

