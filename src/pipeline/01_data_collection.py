from IPython.display import display
#!/usr/bin/env python
# coding: utf-8

#
# # MarketPulse AI
# ## Stock Markets Analytics Zoomcamp 2026 — Capstone Project
#
# ### Notebook 01: Data Collection
#
# **Objective:** Collect historical stock market data and prepare
# the foundation for an end-to-end machine learning trading strategy.
#
# ### Data Sources
# 1. Yahoo Finance — Historical stock prices and trading volume
# 2. FRED — Macroeconomic indicators
# 3. Alternative Data API — Additional market signals
#
# ### Workflow
# 1. Install and import dependencies
# 2. Define stock universe and date range
# 3. Download historical market data
# 4. Inspect data quality and completeness
# 5. Save datasets locally for further processing
#

# In[1]:


# Cell 1

# Install required packages
# Dependencies are installed through requirements.txt


# In[2]:


# Cell 2 - Import Libraries

# Standard libraries
from pathlib import Path
from datetime import datetime

# Data manipulation
import pandas as pd
import numpy as np

# Market data
import yfinance as yf
import requests

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

print("MarketPulse AI - Libraries imported successfully!")
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Yahoo Finance version:", yf.__version__)


# In[3]:


# Cell 3 - Stock Universe and Date Range

# Technology
tech = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

# Financials
financials = ["JPM", "BAC", "GS"]

# Healthcare
healthcare = ["JNJ", "UNH", "PFE"]

# Consumer
consumer = ["WMT", "COST", "HD"]

# Energy
energy = ["XOM", "CVX"]

# Industrials
industrials = ["CAT", "GE"]

# Communications
communications = ["META", "NFLX"]

# Combine stock tickers
tickers = (
    tech + financials + healthcare +
    consumer + energy + industrials +
    communications
)

# Historical period
start_date = "2018-01-01"
end_date = (pd.Timestamp.now(tz="America/New_York") + pd.Timedelta(days=1)).date().isoformat()

# Market benchmark
benchmark = "SPY"

print("Total stocks:", len(tickers))
print("Start date:", start_date)
print("End date:", end_date)
print("Benchmark:", benchmark)
print("Stock tickers:", tickers)


# In[4]:


# Cell 4 - Download Historical Stock Prices

# Include SPY benchmark
all_tickers = tickers + [benchmark]

print(f"Downloading data for {len(all_tickers)} tickers...")
print(f"Period: {start_date} to {end_date}")

# Download historical daily prices
stock_data = yf.download(
    tickers=all_tickers,
    start=start_date,
    end=end_date,
    interval="1d",
    auto_adjust=True,
    group_by="ticker",
    threads=True,
    progress=True
)

# Display download results
print("\nDownload completed!")
print("Dataset shape:", stock_data.shape)
print("Date range:", stock_data.index.min(), "to", stock_data.index.max())

display(stock_data.head())


# In[5]:


# Cell 5 - Data Quality and Missing Values

# Check dataset dimensions
print("Dataset shape:", stock_data.shape)

# Check missing values for each ticker
missing_by_ticker = (
    stock_data.isna()
    .T.groupby(level=0)
    .sum()
    .sum(axis=1)
    .sort_values(ascending=False)
)

print("\nMissing values by ticker:")
display(missing_by_ticker.to_frame("Missing Values"))

# Check duplicate dates
print("\nDuplicate dates:", stock_data.index.duplicated().sum())

# Check total missing values
print("Total missing values:", stock_data.isna().sum().sum())

# Check data types
print("\nData types:")
print(stock_data.dtypes.value_counts())


# In[6]:


# Cell 6 - Convert Stock Data to Long Format

# Convert multi-index columns into rows
stock_long = (
    stock_data
    .stack(level="Ticker", future_stack=True)
    .reset_index()
)

# Standardize column names
stock_long.columns.name = None
stock_long = stock_long.rename(columns={
    "Date": "date",
    "Ticker": "ticker",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume"
})

# Sort by ticker and date
stock_long = (
    stock_long
    .sort_values(["ticker", "date"])
    .reset_index(drop=True)
)

print("Long-format dataset shape:", stock_long.shape)
print("Unique tickers:", stock_long["ticker"].nunique())
print("Date range:", stock_long["date"].min(),
      "to", stock_long["date"].max())

display(stock_long.head(10))


# In[7]:


# Cell 7 - Save Historical Stock Data

# Confirm Jupyter working directory
print("Jupyter working directory:", Path.cwd())

# Locate the project root
project_root = Path.cwd()

if project_root.name == "notebooks":
    project_root = project_root.parent

# Create project data directory
data_dir = project_root / "data"
data_dir.mkdir(parents=True, exist_ok=True)

# Save dataset in the correct location
output_file = data_dir / "stock_prices.parquet"
stock_long.to_parquet(output_file, index=False)

# Verify
stock_check = pd.read_parquet(output_file)

print("\nCorrected file location:", output_file)
print("Reloaded shape:", stock_check.shape)
print("Missing values:", stock_check.isna().sum().sum())


# In[8]:


# Cell 8 - Download FRED Macroeconomic Data

from io import StringIO

fred_series = {
    "DGS10": "treasury_10y",
    "DGS2": "treasury_2y",
    "FEDFUNDS": "fed_funds_rate",
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment_rate"
}

fred_data = None

for series_id, column_name in fred_series.items():
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    df.columns = ["date", column_name]
    df["date"] = pd.to_datetime(df["date"])
    df[column_name] = pd.to_numeric(
        df[column_name], errors="coerce"
    )

    if fred_data is None:
        fred_data = df
    else:
        fred_data = fred_data.merge(
            df, on="date", how="outer"
        )

    print(f"Downloaded {series_id}: {column_name}")

fred_data = (
    fred_data
    .sort_values("date")
    .reset_index(drop=True)
)

print("\nFRED dataset shape:", fred_data.shape)
print("Date range:", fred_data["date"].min(),
      "to", fred_data["date"].max())

display(fred_data.tail(10))


# In[9]:


# Cell 9 - Inspect FRED Data Completeness

# Restrict to our stock market analysis period
fred_filtered = fred_data[
    (fred_data["date"] >= start_date) &
    (fred_data["date"] < end_date)
].copy()

print("Filtered FRED shape:", fred_filtered.shape)

print("\nMissing values by indicator:")
display(
    fred_filtered.isna().sum().to_frame("Missing Values")
)

print("\nNon-null observations by indicator:")
display(
    fred_filtered.notna().sum().to_frame("Observations")
)

print("\nLatest available observation for each indicator:")

for column in fred_series.values():
    valid = fred_filtered.dropna(subset=[column])
    if not valid.empty:
        latest = valid.iloc[-1]
        print(
            f"{column}: "
            f"{latest['date'].date()} = {latest[column]}"
        )
    else:
        print(f"{column}: No observations available")


# In[10]:


# Cell 10 - Align FRED Data to Trading Dates (Corrected)

# Ensure trading dates use pandas datetime64[ns]
trading_dates = pd.DataFrame({
    "date": pd.to_datetime(
        sorted(stock_long["date"].unique())
    )
})

release_delays = {
    "treasury_10y": 1,
    "treasury_2y": 1,
    "fed_funds_rate": 7,
    "cpi": 30,
    "unemployment_rate": 30
}

fred_aligned = trading_dates.copy()

for column, delay in release_delays.items():

    observations = (
        fred_filtered[["date", column]]
        .dropna()
        .copy()
    )

    # Ensure matching datetime types
    observations["date"] = pd.to_datetime(
        observations["date"]
    )

    observations["available_date"] = (
        observations["date"] +
        pd.Timedelta(days=delay)
    ).astype("datetime64[ns]")

    observations = (
        observations[["available_date", column]]
        .sort_values("available_date")
    )

    fred_aligned["date"] = pd.to_datetime(
        fred_aligned["date"]
    ).astype("datetime64[ns]")

    fred_aligned = pd.merge_asof(
        fred_aligned.sort_values("date"),
        observations,
        left_on="date",
        right_on="available_date",
        direction="backward"
    ).drop(columns="available_date")

print("Aligned FRED shape:", fred_aligned.shape)

print("\nMissing values:")
print(fred_aligned.isna().sum())

display(fred_aligned.tail(10))


# In[11]:


# Cell 11 - Clean Aligned FRED Data

# Remove rows with unavailable economic indicators
fred_clean = (
    fred_aligned
    .dropna()
    .reset_index(drop=True)
)

print("Original aligned shape:", fred_aligned.shape)
print("Cleaned FRED shape:", fred_clean.shape)

print("\nRemoved rows:",
      len(fred_aligned) - len(fred_clean))

print("\nCleaned date range:")
print("Start:", fred_clean["date"].min())
print("End:", fred_clean["date"].max())

print("\nRemaining missing values:")
print(fred_clean.isna().sum())

display(fred_clean.head())


# In[12]:


# Cell 12 - Save Cleaned FRED Data

# Use the existing project data directory
output_file = data_dir / "fred_macro.parquet"

# Save as Parquet
fred_clean.to_parquet(
    output_file,
    index=False
)

# Reload to verify
fred_check = pd.read_parquet(output_file)

print("FRED dataset saved successfully!")
print("File:", output_file)
print("File size (MB):",
      round(output_file.stat().st_size / (1024**2), 2))

print("\nReloaded shape:", fred_check.shape)
print("Missing values:", fred_check.isna().sum().sum())
print("Date range:",
      fred_check["date"].min(),
      "to",
      fred_check["date"].max())


# In[13]:


# Cell 13 - Download Alternative Market Sentiment Data

# Alternative.me Crypto Fear & Greed Index API
url = "https://api.alternative.me/fng/"

params = {
    "limit": 0,
    "format": "json"
}

response = requests.get(
    url,
    params=params,
    timeout=30
)

response.raise_for_status()

# Extract API records
sentiment_json = response.json()
sentiment_data = pd.DataFrame(sentiment_json["data"])

# Convert fields to appropriate types
sentiment_data["date"] = pd.to_datetime(
    pd.to_numeric(sentiment_data["timestamp"]),
    unit="s",
    utc=True
).dt.tz_localize(None).dt.normalize()

sentiment_data["fear_greed_index"] = pd.to_numeric(
    sentiment_data["value"],
    errors="coerce"
)

# Keep relevant columns
sentiment_data = sentiment_data[
    ["date", "fear_greed_index", "value_classification"]
]

# Restrict to project period
sentiment_data = sentiment_data[
    (sentiment_data["date"] >= start_date) &
    (sentiment_data["date"] < end_date)
]

sentiment_data = (
    sentiment_data
    .sort_values("date")
    .reset_index(drop=True)
)

print("Sentiment dataset shape:", sentiment_data.shape)
print("Date range:",
      sentiment_data["date"].min(),
      "to",
      sentiment_data["date"].max())

print("\nMissing values:")
print(sentiment_data.isna().sum())

display(sentiment_data.tail(10))


# In[14]:


# Cell 14 - Align Sentiment Data to Trading Dates (Corrected)

# Ensure trading dates have consistent datetime types
trading_dates = pd.DataFrame({
    "date": pd.to_datetime(
        sorted(stock_long["date"].unique())
    )
})

trading_dates["date"] = trading_dates["date"].astype(
    "datetime64[ns]"
)

# Prepare sentiment observations
sentiment_observations = sentiment_data.copy()

sentiment_observations["date"] = pd.to_datetime(
    sentiment_observations["date"]
).astype("datetime64[ns]")

# Apply one-calendar-day availability lag
sentiment_observations["available_date"] = (
    sentiment_observations["date"] +
    pd.Timedelta(days=1)
).astype("datetime64[ns]")

# Prepare columns for alignment
sentiment_observations = (
    sentiment_observations[
        ["available_date",
         "fear_greed_index",
         "value_classification"]
    ]
    .sort_values("available_date")
)

# Align sentiment with trading dates
sentiment_aligned = pd.merge_asof(
    trading_dates.sort_values("date"),
    sentiment_observations,
    left_on="date",
    right_on="available_date",
    direction="backward"
).drop(columns="available_date")

print("Aligned sentiment shape:", sentiment_aligned.shape)

print("\nMissing values:")
print(sentiment_aligned.isna().sum())

print("\nDate range:")
print(
    sentiment_aligned["date"].min(),
    "to",
    sentiment_aligned["date"].max()
)

display(sentiment_aligned.tail(10))


# In[15]:


# Cell 15 - Clean and Save Sentiment Data

# Remove initial missing sentiment observations
sentiment_clean = (
    sentiment_aligned
    .dropna()
    .reset_index(drop=True)
)

# Ensure index values are integers
sentiment_clean["fear_greed_index"] = (
    sentiment_clean["fear_greed_index"].astype(int)
)

# Save dataset to project data directory
sentiment_file = data_dir / "market_sentiment.parquet"

sentiment_clean.to_parquet(
    sentiment_file,
    index=False
)

# Reload to verify
sentiment_check = pd.read_parquet(sentiment_file)

print("Sentiment dataset saved successfully!")
print("File:", sentiment_file)
print("File size (MB):",
      round(sentiment_file.stat().st_size / (1024**2), 2))

print("\nOriginal aligned shape:", sentiment_aligned.shape)
print("Cleaned shape:", sentiment_check.shape)
print("Removed rows:",
      len(sentiment_aligned) - len(sentiment_check))

print("\nDate range:",
      sentiment_check["date"].min(),
      "to",
      sentiment_check["date"].max())

print("Missing values:", sentiment_check.isna().sum().sum())

display(sentiment_check.head())


# In[16]:


# Cell 16 - Merge All Three Data Sources

# Load saved datasets
stocks = pd.read_parquet(data_dir / "stock_prices.parquet")
macro = pd.read_parquet(data_dir / "fred_macro.parquet")
sentiment = pd.read_parquet(data_dir / "market_sentiment.parquet")

# Ensure consistent datetime types
for df in [stocks, macro, sentiment]:
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")

# Merge stock prices with macroeconomic indicators
merged_data = stocks.merge(
    macro,
    on="date",
    how="inner",
    validate="many_to_one"
)

# Merge with market sentiment
merged_data = merged_data.merge(
    sentiment,
    on="date",
    how="inner",
    validate="many_to_one"
)

# Sort records
merged_data = (
    merged_data
    .sort_values(["ticker", "date"])
    .reset_index(drop=True)
)

print("Merged dataset shape:", merged_data.shape)
print("Unique tickers:", merged_data["ticker"].nunique())
print("Unique trading dates:", merged_data["date"].nunique())

print("\nDate range:")
print(merged_data["date"].min(),
      "to", merged_data["date"].max())

print("\nMissing values:",
      merged_data.isna().sum().sum())

print("\nColumns:")
print(merged_data.columns.tolist())

display(merged_data.head())


# In[17]:


# Cell 17 - Save Merged Dataset

# Define output path
merged_file = data_dir / "marketpulse_merged.parquet"

# Save merged dataset
merged_data.to_parquet(
    merged_file,
    index=False
)

# Reload to verify
merged_check = pd.read_parquet(merged_file)

print("Merged dataset saved successfully!")
print("File:", merged_file)
print("File size (MB):",
      round(merged_file.stat().st_size / (1024**2), 2))

print("\nReloaded shape:", merged_check.shape)
print("Unique tickers:", merged_check["ticker"].nunique())
print("Unique trading dates:", merged_check["date"].nunique())
print("Missing values:", merged_check.isna().sum().sum())

print("\nDate range:")
print(merged_check["date"].min(),
      "to", merged_check["date"].max())
