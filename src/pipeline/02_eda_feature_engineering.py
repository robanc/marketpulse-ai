from IPython.display import display
#!/usr/bin/env python
# coding: utf-8

#
# # MarketPulse AI
# ## Stock Markets Analytics Zoomcamp 2026 — Capstone Project
#
# ### Notebook 02: Exploratory Data Analysis and Feature Engineering
#
# **Objective:** Explore the merged market dataset and develop
# predictive features for stock return forecasting and trading
# strategy evaluation.
#
# ### Data Sources
# 1. Yahoo Finance — Historical Stock Prices
# 2. FRED — Macroeconomic Indicators
# 3. Alternative.me — Crypto Fear & Greed Index
#
# ### Workflow
# 1. Load and validate the merged dataset
# 2. Exploratory Data Analysis (EDA)
# 3. Analyze stock returns and volatility
# 4. Engineer 20+ predictive features
# 5. Define 10-trading-day forward return targets
# 6. Validate feature quality and prevent data leakage
# 7. Save the feature-engineered dataset
#
# ### Modeling Objective
# Predict 10-trading-day forward stock returns and evaluate
# whether predictions can support a profitable trading strategy
# after transaction costs.
#
# ### Important Considerations
# - Use chronological training and validation splits.
# - Avoid future information in predictive features.
# - Treat SPY as the benchmark, not a trading candidate.
# - Validate economic indicator publication timing and revisions
#   before final backtesting.
#

# In[1]:


# Cell 1 - Import Libraries and Load Dataset

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

# Project directory
project_root = Path.cwd()

if project_root.name == "notebooks":
    project_root = project_root.parent

data_dir = project_root / "data"

# Load merged dataset
data_file = data_dir / "marketpulse_merged.parquet"
df = pd.read_parquet(data_file)

# Ensure proper datetime format
df["date"] = pd.to_datetime(df["date"])

# Sort dataset
df = df.sort_values(
    ["ticker", "date"]
).reset_index(drop=True)

print("MarketPulse AI - Notebook 02")
print("Dataset loaded successfully!")

print("\nDataset shape:", df.shape)
print("Unique tickers:", df["ticker"].nunique())
print("Trading dates:", df["date"].nunique())

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())

print("\nMissing values:", df.isna().sum().sum())

display(df.head())


# In[2]:


# Cell 2 - Dataset Overview and Descriptive Statistics

print("DATASET INFORMATION")
print("-" * 50)

df.info()

print("\nNUMERICAL SUMMARY")
print("-" * 50)

display(
    df.describe().round(2)
)

print("\nRECORDS PER TICKER")
print("-" * 50)

ticker_counts = (
    df.groupby("ticker")
    .size()
    .sort_values(ascending=False)
)

display(
    ticker_counts.to_frame("record_count")
)

print("\nSENTIMENT CLASSIFICATION DISTRIBUTION")
print("-" * 50)

sentiment_distribution = (
    df.drop_duplicates("date")["value_classification"]
    .value_counts()
)

display(
    sentiment_distribution.to_frame("trading_days")
)


# In[3]:


# Cell 3 - Visualize Normalized Stock Prices

selected_tickers = [
    "AAPL", "MSFT", "NVDA",
    "GOOGL", "AMZN", "SPY"
]

# Filter selected stocks
price_data = df[
    df["ticker"].isin(selected_tickers)
].copy()

# Pivot into wide format
price_pivot = price_data.pivot(
    index="date",
    columns="ticker",
    values="close"
)

# Normalize each stock to 100 at the start
normalized_prices = (
    price_pivot / price_pivot.iloc[0] * 100
)

# Plot normalized prices
plt.figure(figsize=(14, 7))

for ticker in selected_tickers:
    plt.plot(
        normalized_prices.index,
        normalized_prices[ticker],
        label=ticker,
        linewidth=1.5
    )

plt.title(
    "MarketPulse AI - Normalized Stock Price Performance",
    fontsize=15
)
plt.xlabel("Date")
plt.ylabel("Normalized Price (Start = 100)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("Normalized price dataset shape:",
      normalized_prices.shape)

display(normalized_prices.tail())


# In[4]:


# Cell 4 - Daily Returns and Volatility Analysis

# Create a working copy
eda_df = df.copy()

# Calculate daily returns separately for each ticker
eda_df["daily_return"] = (
    eda_df.groupby("ticker")["close"]
    .pct_change()
)

# Calculate annualized volatility
# Assumes approximately 252 trading days per year
volatility_summary = (
    eda_df.groupby("ticker")["daily_return"]
    .agg(["mean", "std"])
)

volatility_summary["annualized_return_approx"] = (
    volatility_summary["mean"] * 252 * 100
)

volatility_summary["annualized_volatility"] = (
    volatility_summary["std"] * np.sqrt(252) * 100
)

# Display percentage values
volatility_summary["mean_daily_return_pct"] = (
    volatility_summary["mean"] * 100
)

volatility_summary = volatility_summary[
    [
        "mean_daily_return_pct",
        "annualized_return_approx",
        "annualized_volatility"
    ]
].sort_values(
    "annualized_volatility",
    ascending=False
)

print("DAILY RETURNS AND VOLATILITY SUMMARY")
print("-" * 50)

display(volatility_summary.round(2))

# Visualize annualized volatility
plt.figure(figsize=(13, 7))

sns.barplot(
    data=volatility_summary.reset_index(),
    x="annualized_volatility",
    y="ticker"
)

plt.title(
    "MarketPulse AI - Annualized Historical Volatility"
)
plt.xlabel("Annualized Volatility (%)")
plt.ylabel("Ticker")
plt.tight_layout()
plt.show()

print("\nMissing daily returns:",
      eda_df["daily_return"].isna().sum())


# In[5]:


# Cell 5 - Stock Return Correlation Analysis

# Convert daily returns into wide format
returns_pivot = eda_df.pivot(
    index="date",
    columns="ticker",
    values="daily_return"
)

# Calculate correlation matrix
correlation_matrix = returns_pivot.corr()

print("Correlation matrix shape:",
      correlation_matrix.shape)

# Plot correlation heatmap
plt.figure(figsize=(14, 11))

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.3,
    annot_kws={"size": 7}
)

plt.title(
    "MarketPulse AI - Daily Stock Return Correlation",
    fontsize=15
)

plt.tight_layout()
plt.show()

# Identify strongest and weakest stock correlations

# Exclude SPY benchmark
stock_corr = correlation_matrix.drop(
    index="SPY",
    columns="SPY"
)

# Extract unique stock pairs from upper triangle
upper_triangle = np.triu(
    np.ones(stock_corr.shape, dtype=bool),
    k=1
)

pairs = (
    stock_corr
    .where(upper_triangle)
    .stack()
    .dropna()
    .sort_values(ascending=False)
)

print("\nTotal unique stock pairs:", len(pairs))

print("\nTop 10 positively correlated stock pairs:")
display(
    pairs.head(10).to_frame("correlation")
)

print("\nTop 10 least correlated stock pairs:")
display(
    pairs.tail(10).sort_values().to_frame("correlation")
)


# In[6]:


# Cell 6 - Maximum Drawdown Analysis

# Create working copy
drawdown_df = df[["date", "ticker", "close"]].copy()

# Calculate running peak for each ticker
drawdown_df["running_peak"] = (
    drawdown_df.groupby("ticker")["close"]
    .cummax()
)

# Calculate drawdown percentage
drawdown_df["drawdown_pct"] = (
    drawdown_df["close"] /
    drawdown_df["running_peak"] - 1
) * 100

# Calculate maximum drawdown for each ticker
max_drawdown = (
    drawdown_df.groupby("ticker")["drawdown_pct"]
    .min()
    .sort_values()
)

print("MAXIMUM DRAWDOWN SUMMARY")
print("-" * 50)

display(
    max_drawdown.to_frame("max_drawdown_pct").round(2)
)

# Plot maximum drawdowns
plt.figure(figsize=(13, 7))

sns.barplot(
    x=max_drawdown.values,
    y=max_drawdown.index
)

plt.title(
    "MarketPulse AI - Maximum Historical Drawdown"
)
plt.xlabel("Maximum Drawdown (%)")
plt.ylabel("Ticker")
plt.tight_layout()
plt.show()

# Compare selected stocks with SPY
selected_drawdowns = drawdown_df[
    drawdown_df["ticker"].isin(
        ["AAPL", "MSFT", "NVDA", "SPY"]
    )
]

plt.figure(figsize=(14, 6))

for ticker in ["AAPL", "MSFT", "NVDA", "SPY"]:
    subset = selected_drawdowns[
        selected_drawdowns["ticker"] == ticker
    ]

    plt.plot(
        subset["date"],
        subset["drawdown_pct"],
        label=ticker
    )

plt.title(
    "MarketPulse AI - Historical Drawdown Comparison"
)
plt.xlabel("Date")
plt.ylabel("Drawdown from Previous Peak (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# In[7]:


# Cell 7 - Market Sentiment and SPY Returns

# Select SPY benchmark data
spy_df = eda_df[
    eda_df["ticker"] == "SPY"
].copy()

# Remove the first missing daily return
spy_df = spy_df.dropna(
    subset=["daily_return"]
)

# Convert returns to percentage
spy_df["daily_return_pct"] = (
    spy_df["daily_return"] * 100
)

# Define sentiment category order
sentiment_order = [
    "Extreme Fear",
    "Fear",
    "Neutral",
    "Greed",
    "Extreme Greed"
]

# Calculate summary statistics
sentiment_returns = (
    spy_df.groupby("value_classification")[
        "daily_return_pct"
    ]
    .agg(["count", "mean", "median", "std"])
    .reindex(sentiment_order)
)

print("SPY RETURNS BY SENTIMENT CLASSIFICATION")
print("-" * 50)

display(sentiment_returns.round(3))

# Plot return distributions
plt.figure(figsize=(12, 6))

sns.boxplot(
    data=spy_df,
    x="value_classification",
    y="daily_return_pct",
    order=sentiment_order,
    showfliers=False
)

plt.title(
    "MarketPulse AI - SPY Daily Returns by Crypto Sentiment"
)
plt.xlabel("Crypto Fear & Greed Classification")
plt.ylabel("SPY Daily Return (%)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate correlation
sentiment_corr = spy_df[
    ["fear_greed_index", "daily_return"]
].corr().iloc[0, 1]

print(
    "\nCorrelation between Crypto Fear & Greed "
    "and SPY daily returns:",
    round(sentiment_corr, 4)
)


# In[8]:


# Cell 8 - Macroeconomic Indicators and SPY Returns

macro_columns = [
    "treasury_10y",
    "treasury_2y",
    "fed_funds_rate",
    "cpi",
    "unemployment_rate"
]

# Use one row per trading day (SPY only)
spy_macro = eda_df[
    eda_df["ticker"] == "SPY"
].copy()

# Calculate daily SPY returns in percentage
spy_macro["spy_return_pct"] = (
    spy_macro["daily_return"] * 100
)

# Correlation matrix
macro_corr = spy_macro[
    macro_columns + ["spy_return_pct"]
].corr()

print("MACROECONOMIC CORRELATION MATRIX")
print("-" * 50)

display(macro_corr.round(3))

# Correlation with SPY daily returns
spy_macro_corr = (
    macro_corr["spy_return_pct"]
    .drop("spy_return_pct")
    .sort_values(ascending=False)
)

print("\nCORRELATION WITH SPY DAILY RETURNS")
print("-" * 50)

display(
    spy_macro_corr.to_frame("correlation").round(4)
)

# Plot heatmap
plt.figure(figsize=(10, 7))

sns.heatmap(
    macro_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1
)

plt.title(
    "MarketPulse AI - Macroeconomic Indicators and SPY Returns"
)
plt.tight_layout()
plt.show()


# In[9]:


# Cell 9 - Feature Engineering: Price Momentum

# Create feature engineering dataset
features_df = df.copy()

# Ensure correct ordering
features_df = (
    features_df
    .sort_values(["ticker", "date"])
    .reset_index(drop=True)
)

# Group data by ticker
grouped = features_df.groupby("ticker")["close"]

# Calculate historical price momentum
features_df["return_1d"] = grouped.pct_change(1)
features_df["return_5d"] = grouped.pct_change(5)
features_df["return_10d"] = grouped.pct_change(10)
features_df["return_20d"] = grouped.pct_change(20)

# Display new features
momentum_columns = [
    "ticker",
    "date",
    "close",
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d"
]

print("PRICE MOMENTUM FEATURES")
print("-" * 50)

display(
    features_df[momentum_columns].head(25)
)

print("\nMissing values in momentum features:")
print(
    features_df[
        ["return_1d", "return_5d",
         "return_10d", "return_20d"]
    ].isna().sum()
)

print("\nFeature engineering dataset shape:",
      features_df.shape)


# In[10]:


# Cell 10 - Feature Engineering: Moving Averages

# Simple Moving Averages (SMA)
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

# Exponential Moving Averages (EMA)
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

# Display new features
ma_columns = [
    "ticker",
    "date",
    "close",
    "sma_5",
    "sma_10",
    "sma_20",
    "sma_50",
    "ema_12",
    "ema_26"
]

print("MOVING AVERAGE FEATURES")
print("-" * 50)

display(
    features_df[ma_columns].iloc[45:55]
)

print("\nMissing values in moving averages:")
print(
    features_df[
        ["sma_5", "sma_10", "sma_20",
         "sma_50", "ema_12", "ema_26"]
    ].isna().sum()
)

print("\nFeature engineering dataset shape:",
      features_df.shape)


# In[11]:


# Cell 11 - Feature Engineering: Volatility Features

# Rolling annualized volatility from daily returns
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

# Daily high-low range relative to closing price
features_df["daily_range_pct"] = (
    (features_df["high"] - features_df["low"]) /
    features_df["close"]
)

# Previous closing price for each ticker
previous_close = (
    features_df.groupby("ticker")["close"].shift(1)
)

# True Range components
true_range = pd.concat(
    [
        features_df["high"] - features_df["low"],
        (features_df["high"] - previous_close).abs(),
        (features_df["low"] - previous_close).abs()
    ],
    axis=1
).max(axis=1)

# Average True Range over 14 trading days
features_df["atr_14"] = (
    true_range.groupby(features_df["ticker"])
    .transform(
        lambda x: x.rolling(
            window=14,
            min_periods=14
        ).mean()
    )
)

# Display results
volatility_columns = [
    "ticker",
    "date",
    "close",
    "volatility_5d",
    "volatility_20d",
    "daily_range_pct",
    "atr_14"
]

print("VOLATILITY FEATURES")
print("-" * 50)

display(
    features_df[volatility_columns].iloc[20:30]
)

print("\nMissing values in volatility features:")
print(
    features_df[
        [
            "volatility_5d",
            "volatility_20d",
            "daily_range_pct",
            "atr_14"
        ]
    ].isna().sum()
)

print("\nFeature engineering dataset shape:",
      features_df.shape)


# In[12]:


# Cell 12 - Feature Engineering: Technical Indicators

# 1. Relative Strength Index (RSI 14)
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

# Handle periods with no losses or no price changes
features_df.loc[
    (avg_loss == 0) & (avg_gain > 0), "rsi_14"
] = 100

features_df.loc[
    (avg_loss == 0) & (avg_gain == 0), "rsi_14"
] = 50

# 2. MACD
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

# 3. Bollinger Bands (20-day SMA +/- 2 standard deviations)
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

# Display new features
technical_columns = [
    "ticker",
    "date",
    "close",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bollinger_upper",
    "bollinger_lower"
]

print("TECHNICAL INDICATOR FEATURES")
print("-" * 50)

display(
    features_df[technical_columns].iloc[30:40].round(4)
)

print("\nMissing values in technical indicators:")
print(
    features_df[
        [
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_histogram",
            "bollinger_upper",
            "bollinger_lower"
        ]
    ].isna().sum()
)

print("\nFeature engineering dataset shape:",
      features_df.shape)


# In[13]:


# Cell 13 - Feature Engineering: Volume Features

# Daily percentage change in trading volume
features_df["volume_change_1d"] = (
    features_df.groupby("ticker")["volume"]
    .pct_change()
)

# 20-day average trading volume
features_df["volume_sma_20"] = (
    features_df.groupby("ticker")["volume"]
    .transform(
        lambda x: x.rolling(
            window=20,
            min_periods=20
        ).mean()
    )
)

# Relative trading volume
features_df["relative_volume_20"] = (
    features_df["volume"] /
    features_df["volume_sma_20"]
)

# Logarithmic volume transformation
features_df["log_volume"] = np.log1p(
    features_df["volume"]
)

# Display new features
volume_columns = [
    "ticker",
    "date",
    "volume",
    "volume_change_1d",
    "volume_sma_20",
    "relative_volume_20",
    "log_volume"
]

print("VOLUME FEATURES")
print("-" * 50)

display(
    features_df[volume_columns].iloc[20:30]
    .round({
        "volume_change_1d": 4,
        "volume_sma_20": 2,
        "relative_volume_20": 4,
        "log_volume": 4
    })
)

print("\nMissing values in volume features:")
print(
    features_df[
        [
            "volume_change_1d",
            "volume_sma_20",
            "relative_volume_20",
            "log_volume"
        ]
    ].isna().sum()
)

print("\nFeature engineering dataset shape:",
      features_df.shape)


# In[14]:


# Cell 14 - Feature Engineering: Macroeconomic Features

# Create one macroeconomic observation per trading day
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

# Treasury yield curve spread (percentage points)
macro_daily["yield_spread"] = (
    macro_daily["treasury_10y"] -
    macro_daily["treasury_2y"]
)

# Five-trading-day changes in Treasury yields
macro_daily["treasury_10y_change_5d"] = (
    macro_daily["treasury_10y"].diff(5)
)

macro_daily["treasury_2y_change_5d"] = (
    macro_daily["treasury_2y"].diff(5)
)

# Twenty-trading-day change in federal funds rate
macro_daily["fed_funds_change_20d"] = (
    macro_daily["fed_funds_rate"].diff(20)
)

# Sixty-trading-day percentage change in CPI
macro_daily["cpi_change_60d"] = (
    macro_daily["cpi"].pct_change(60)
)

# Select new features for merging
macro_feature_columns = [
    "yield_spread",
    "treasury_10y_change_5d",
    "treasury_2y_change_5d",
    "fed_funds_change_20d",
    "cpi_change_60d"
]

# Merge macro features back to stock observations
features_df = features_df.merge(
    macro_daily[["date"] + macro_feature_columns],
    on="date",
    how="left",
    validate="many_to_one"
)

# Display results
print("MACROECONOMIC FEATURES")
print("-" * 50)

display(
    macro_daily[
        ["date"] + macro_feature_columns
    ].iloc[60:70].round({
        "yield_spread": 4,
        "treasury_10y_change_5d": 4,
        "treasury_2y_change_5d": 4,
        "fed_funds_change_20d": 4,
        "cpi_change_60d": 6
    })
)

print("\nMissing values in macroeconomic features:")
print(
    features_df[macro_feature_columns].isna().sum()
)

print("\nFeature engineering dataset shape:",
      features_df.shape)


# In[15]:


# Cell 15 - Feature Engineering: Sentiment Features

# One sentiment observation per trading date
sentiment_daily = (
    features_df[
        ["date", "fear_greed_index"]
    ]
    .drop_duplicates(subset="date")
    .sort_values("date")
    .reset_index(drop=True)
)

# Change in sentiment since previous trading day
sentiment_daily["sentiment_change_1d"] = (
    sentiment_daily["fear_greed_index"].diff(1)
)

# Rolling sentiment averages
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

# Rolling sentiment volatility
sentiment_daily["sentiment_volatility_20"] = (
    sentiment_daily["fear_greed_index"]
    .rolling(window=20, min_periods=20)
    .std()
)

# New sentiment feature columns
sentiment_feature_columns = [
    "sentiment_change_1d",
    "sentiment_sma_5",
    "sentiment_sma_20",
    "sentiment_volatility_20"
]

# Merge sentiment features back to all tickers
features_df = features_df.merge(
    sentiment_daily[
        ["date"] + sentiment_feature_columns
    ],
    on="date",
    how="left",
    validate="many_to_one"
)

print("SENTIMENT FEATURES")
print("-" * 50)

display(
    sentiment_daily[
        ["date", "fear_greed_index"] +
        sentiment_feature_columns
    ].iloc[20:30].round({
        "sentiment_sma_5": 2,
        "sentiment_sma_20": 2,
        "sentiment_volatility_20": 2
    })
)

print("\nMissing values in sentiment features:")
print(
    features_df[
        sentiment_feature_columns
    ].isna().sum()
)

print("\nFeature engineering dataset shape:",
      features_df.shape)


# In[16]:


# Cell 16 - Create 10-Trading-Day Forward Return Targets

# Ensure chronological order within each ticker
features_df = (
    features_df
    .sort_values(["ticker", "date"])
    .reset_index(drop=True)
)

# Closing price 10 trading days into the future
features_df["future_close_10d"] = (
    features_df.groupby("ticker")["close"]
    .shift(-10)
)

# Regression target: forward 10-day percentage return
features_df["target_return_10d"] = (
    features_df["future_close_10d"] /
    features_df["close"] - 1
)

# Classification target: positive or non-positive return
# Preserve missing targets at the end of each ticker
features_df["target_direction_10d"] = (
    features_df["target_return_10d"]
    .gt(0)
    .astype("Int64")
    .where(features_df["target_return_10d"].notna())
)

# Display sample targets
target_columns = [
    "ticker",
    "date",
    "close",
    "future_close_10d",
    "target_return_10d",
    "target_direction_10d"
]

print("10-DAY FORWARD RETURN TARGETS")
print("-" * 50)

display(
    features_df[target_columns]
    .iloc[45:55]
    .round({
        "close": 4,
        "future_close_10d": 4,
        "target_return_10d": 6
    })
)

print("\nMissing values in targets:")
print(
    features_df[
        ["target_return_10d", "target_direction_10d"]
    ].isna().sum()
)

print("\nTarget direction distribution:")
print(
    features_df["target_direction_10d"]
    .value_counts(dropna=False)
)

print("\nFeature engineering dataset shape:",
      features_df.shape)


# In[17]:


# Cell 17 - Feature Quality Checks

# Explicitly identify the 33 engineered predictive features
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

print("FEATURE QUALITY CHECKS")
print("-" * 50)

print("Number of engineered features:",
      len(feature_columns))

print("\nDuplicate ticker-date records:",
      features_df.duplicated(
          subset=["ticker", "date"]
      ).sum())

# Count missing feature values
missing_features = (
    features_df[feature_columns]
    .isna()
    .sum()
    .sort_values(ascending=False)
)

print("\nMissing values by feature:")
display(missing_features.to_frame("missing_count"))

# Check for positive or negative infinity
infinite_count = np.isinf(
    features_df[feature_columns].to_numpy(dtype=float)
).sum()

print("\nInfinite feature values:", infinite_count)

# Check rows with all engineered features available
complete_feature_mask = (
    features_df[feature_columns]
    .notna()
    .all(axis=1)
)

print("\nRows with complete engineered features:",
      complete_feature_mask.sum())

print("First complete-feature date:",
      features_df.loc[
          complete_feature_mask, "date"
      ].min())

# Check rows with complete features and known targets
model_ready_mask = (
    complete_feature_mask &
    features_df[target_columns].notna().all(axis=1)
)

print("\nRows with complete features and targets:",
      model_ready_mask.sum())

print("\nDataset shape:", features_df.shape)


# In[18]:



# Cell 17A - Feature-to-Target Correlation Analysis

# Use complete labeled stock observations, excluding SPY.
correlation_df = features_df.loc[
    model_ready_mask & (features_df["ticker"] != "SPY"),
    feature_columns + ["target_return_10d"]
].copy()

feature_target_corr = (
    correlation_df[feature_columns]
    .corrwith(correlation_df["target_return_10d"])
    .dropna()
    .sort_values(ascending=False)
)

print("FEATURE CORRELATIONS WITH 10-DAY FORWARD RETURNS")
print("-" * 55)

print("\nTop 10 positive correlations:")
print(feature_target_corr.head(10).round(4))

print("\nTop 10 negative correlations:")
print(feature_target_corr.tail(10).sort_values().round(4))

plt.figure(figsize=(12, 10))
sns.barplot(
    x=feature_target_corr.values,
    y=feature_target_corr.index,
    color="steelblue"
)
plt.axvline(0, color="black", linewidth=1)
plt.title("Engineered Feature Correlation with 10-Day Forward Return")
plt.xlabel("Pearson Correlation")
plt.ylabel("Engineered Feature")
plt.tight_layout()
plt.show()

correlation_path = (
    project_root / "results" / "feature_target_correlations.csv"
)
correlation_path.parent.mkdir(parents=True, exist_ok=True)

feature_target_corr.rename("correlation").to_csv(
    correlation_path,
    header=True
)

print("\nSaved correlation analysis:", correlation_path)
print("Number of features analyzed:", len(feature_target_corr))


# Cell 18 - Prepare and Save Machine Learning Dataset

from pathlib import Path

data_dir = project_root / "data"
data_dir.mkdir(parents=True, exist_ok=True)

# Keep observations with complete features and targets
model_df = (
    features_df.loc[model_ready_mask]
    .copy()
    .sort_values(["date", "ticker"])
    .reset_index(drop=True)
)

# SPY is reserved for benchmark comparison
training_df = (
    model_df.loc[model_df["ticker"] != "SPY"]
    .copy()
    .reset_index(drop=True)
)

# Remove the future price helper from training data
# It must never be used as a predictive input
training_df = training_df.drop(
    columns=["future_close_10d"]
)

# Save complete engineered dataset for analysis/backtesting
full_path = data_dir / "marketpulse_features.parquet"
features_df.to_parquet(full_path, index=False)

# Save labeled, model-ready stock dataset
training_path = data_dir / "marketpulse_training.parquet"
training_df.to_parquet(training_path, index=False)

print("MACHINE LEARNING DATASET PREPARATION")
print("-" * 50)

print("Full engineered dataset:", features_df.shape)
print("Complete labeled dataset:", model_df.shape)
print("Stock training dataset:", training_df.shape)

print("\nTraining tickers:",
      training_df["ticker"].nunique())

print("SPY included in training:",
      "SPY" in training_df["ticker"].unique())

print("\nTraining date range:")
print("Start:", training_df["date"].min())
print("End:  ", training_df["date"].max())

print("\nMissing engineered features:",
      training_df[feature_columns].isna().sum().sum())

print("Missing targets:",
      training_df[target_columns].isna().sum().sum())

print("\nSaved files:")
print(full_path)
print(training_path)

# Verify saved files
verified_training = pd.read_parquet(training_path)

print("\nVerified saved training shape:",
      verified_training.shape)
