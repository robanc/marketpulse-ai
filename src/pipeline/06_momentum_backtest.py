"""
MarketPulse AI - Alternative 10-Day Momentum Strategy

Independent trading strategy for Criterion 5.
Signals use historical closing prices only.
Entries execute at the next trading session's open.
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRICES_PATH = PROJECT_ROOT / "data" / "marketpulse_merged.parquet"
PREDICTIONS_PATH = (
    PROJECT_ROOT / "results" / "marketpulse_test_predictions.parquet"
)

INITIAL_CAPITAL = 100_000.00
MAX_POSITIONS = 10
POSITION_SIZE = INITIAL_CAPITAL / MAX_POSITIONS

MOMENTUM_LOOKBACK = 10
HOLDING_PERIOD = 10

COMMISSION_RATE = 0.0
SLIPPAGE_RATE = 0.0005
RISK_FREE_RATE = 0.0

print("MarketPulse AI - Momentum Strategy")
print("Project root:", PROJECT_ROOT)
print("Initial capital:", f"${INITIAL_CAPITAL:,.2f}")
print("Maximum positions:", MAX_POSITIONS)
print("Momentum lookback:", MOMENTUM_LOOKBACK)
print("Holding period:", HOLDING_PERIOD)

# Step 139S - Load Data and Generate Momentum Signals

prices = pd.read_parquet(PRICES_PATH)
predictions = pd.read_parquet(PREDICTIONS_PATH)

prices["date"] = pd.to_datetime(prices["date"])
predictions["date"] = pd.to_datetime(predictions["date"])

prices["ticker"] = prices["ticker"].str.upper()
predictions["ticker"] = predictions["ticker"].str.upper()

prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

assert not prices.duplicated(["date", "ticker"]).any()
assert not predictions.duplicated(["date", "ticker"]).any()

# Calculate momentum using current and historical closing prices.
prices["momentum_10d"] = (
    prices.groupby("ticker")["close"]
    .pct_change(MOMENTUM_LOOKBACK)
)

# Match the exact test observations used by MarketPulse AI.
test_keys = predictions[["date", "ticker"]].copy()

momentum_signals = test_keys.merge(
    prices[["date", "ticker", "momentum_10d"]],
    on=["date", "ticker"],
    how="left",
    validate="one_to_one"
)

assert momentum_signals["momentum_10d"].notna().all()

# Predefined eligibility rule: positive trailing momentum.
momentum_signals["signal"] = (
    momentum_signals["momentum_10d"] > 0
).astype(int)

print("\nMOMENTUM SIGNAL VALIDATION")
print("-" * 45)
print("Test observations:", len(momentum_signals))
print("Positive signals:", int(momentum_signals["signal"].sum()))
print("Missing momentum:", momentum_signals["momentum_10d"].isna().sum())
print("Start date:", momentum_signals["date"].min())
print("End date:", momentum_signals["date"].max())

assert len(momentum_signals) == len(predictions)
print("Step 139S verification: PASSED")

# Step 139T - Calculate Trade Entry and Exit Prices

prices["entry_date"] = (
    prices.groupby("ticker")["date"].shift(-1)
)

prices["entry_open"] = (
    prices.groupby("ticker")["open"].shift(-1)
)

prices["exit_date"] = (
    prices.groupby("ticker")["date"].shift(-(HOLDING_PERIOD + 1))
)

prices["exit_open"] = (
    prices.groupby("ticker")["open"].shift(-(HOLDING_PERIOD + 1))
)

momentum_trades = momentum_signals.merge(
    prices[
        [
            "date", "ticker",
            "entry_date", "entry_open",
            "exit_date", "exit_open"
        ]
    ],
    on=["date", "ticker"],
    how="left",
    validate="one_to_one"
)

momentum_trades["valid_trade"] = (
    momentum_trades["entry_open"].notna()
    & momentum_trades["exit_open"].notna()
    & (momentum_trades["entry_open"] > 0)
    & (momentum_trades["exit_open"] > 0)
    & (momentum_trades["exit_date"] > momentum_trades["entry_date"])
)

momentum_candidates = momentum_trades[
    (momentum_trades["signal"] == 1)
    & momentum_trades["valid_trade"]
].copy()

momentum_candidates["gross_return"] = (
    momentum_candidates["exit_open"]
    / momentum_candidates["entry_open"] - 1
)

print("\nMOMENTUM ENTRY/EXIT VALIDATION")
print("-" * 45)
print("Positive momentum signals:", int(momentum_signals["signal"].sum()))
print("Valid trade candidates:", len(momentum_candidates))
print("Missing entry prices:", momentum_trades["entry_open"].isna().sum())
print("Missing exit prices:", momentum_trades["exit_open"].isna().sum())

assert momentum_candidates["valid_trade"].all()
assert (
    momentum_candidates["exit_date"] > momentum_candidates["entry_date"]
).all()

print("Step 139T verification: PASSED")

# Step 139U - Apply Transaction Costs

momentum_candidates["effective_entry_price"] = (
    momentum_candidates["entry_open"] * (1 + SLIPPAGE_RATE)
)

momentum_candidates["effective_exit_price"] = (
    momentum_candidates["exit_open"] * (1 - SLIPPAGE_RATE)
)

momentum_candidates["net_return"] = (
    (
        momentum_candidates["effective_exit_price"]
        * (1 - COMMISSION_RATE)
    )
    /
    (
        momentum_candidates["effective_entry_price"]
        * (1 + COMMISSION_RATE)
    )
    - 1
)

# Rank by entry date, then strongest historical momentum.
momentum_candidates = momentum_candidates.sort_values(
    ["entry_date", "momentum_10d", "ticker"],
    ascending=[True, False, True]
).reset_index(drop=True)

print("\nMOMENTUM TRANSACTION COST VALIDATION")
print("-" * 45)
print("Trade candidates:", len(momentum_candidates))
print("Commission per side:", f"{COMMISSION_RATE:.2%}")
print("Slippage per side:", f"{SLIPPAGE_RATE:.2%}")
print("Average gross return:", f"{momentum_candidates['gross_return'].mean():.4%}")
print("Average net return:", f"{momentum_candidates['net_return'].mean():.4%}")

assert (
    momentum_candidates["net_return"]
    <= momentum_candidates["gross_return"] + 1e-10
).all()

assert momentum_candidates["net_return"].notna().all()

print("Step 139U verification: PASSED")

# Step 139V - Momentum Portfolio Execution

cash = INITIAL_CAPITAL
active_positions = []
executed_trades = []

skipped_overlap = 0
skipped_capacity = 0
skipped_cash = 0

for _, trade in momentum_candidates.iterrows():

    entry_date = trade["entry_date"]
    ticker = trade["ticker"]

    # Release capital from completed positions.
    remaining_positions = []

    for position in active_positions:
        if position["exit_date"] <= entry_date:
            cash += position["exit_value"]
        else:
            remaining_positions.append(position)

    active_positions = remaining_positions

    # Prevent overlapping positions in the same stock.
    active_tickers = {
        position["ticker"] for position in active_positions
    }

    if ticker in active_tickers:
        skipped_overlap += 1
        continue

    # Enforce portfolio capacity.
    if len(active_positions) >= MAX_POSITIONS:
        skipped_capacity += 1
        continue

    # Enforce available cash.
    if cash + 1e-8 < POSITION_SIZE:
        skipped_cash += 1
        continue

    # Execute the trade.
    cash -= POSITION_SIZE

    exit_value = POSITION_SIZE * (1 + trade["net_return"])

    active_positions.append({
        "ticker": ticker,
        "exit_date": trade["exit_date"],
        "exit_value": exit_value
    })

    executed_trades.append({
        "ticker": ticker,
        "signal_date": trade["date"],
        "entry_date": entry_date,
        "exit_date": trade["exit_date"],
        "momentum_10d": trade["momentum_10d"],
        "allocated_capital": POSITION_SIZE,
        "net_return": trade["net_return"],
        "pnl": exit_value - POSITION_SIZE
    })

momentum_executed_df = pd.DataFrame(executed_trades)

realized_pnl = momentum_executed_df["pnl"].sum()
final_equity = INITIAL_CAPITAL + realized_pnl

print("\nMOMENTUM PORTFOLIO EXECUTION")
print("-" * 45)
print("Candidates:", len(momentum_candidates))
print("Executed trades:", len(momentum_executed_df))
print("Skipped overlap:", skipped_overlap)
print("Skipped capacity:", skipped_capacity)
print("Skipped cash:", skipped_cash)
print("Total realized P&L:", f"${realized_pnl:,.2f}")
print("Final equity:", f"${final_equity:,.2f}")
print("Cumulative return:", f"{realized_pnl / INITIAL_CAPITAL:.2%}")

assert (
    len(momentum_executed_df)
    + skipped_overlap
    + skipped_capacity
    + skipped_cash
    == len(momentum_candidates)
)

assert final_equity > 0

for ticker, group in momentum_executed_df.groupby("ticker"):
    ordered = group.sort_values("entry_date")
    assert (
        ordered["entry_date"].iloc[1:].reset_index(drop=True)
        >= ordered["exit_date"].iloc[:-1].reset_index(drop=True)
    ).all(), f"Overlapping trades found for {ticker}"

print("Step 139V verification: PASSED")

# Step 140A - Daily Momentum Portfolio Valuation

benchmark_daily = pd.read_parquet(
    PROJECT_ROOT / "results" / "marketpulse_benchmark_daily.parquet"
)

trading_dates = pd.to_datetime(
    benchmark_daily["date"]
).sort_values().reset_index(drop=True)

close_prices = prices.pivot(
    index="date",
    columns="ticker",
    values="close"
)

# Recover effective execution prices for executed trades.
momentum_positions = momentum_executed_df.merge(
    momentum_candidates[
        [
            "ticker", "entry_date", "exit_date",
            "effective_entry_price", "effective_exit_price"
        ]
    ],
    on=["ticker", "entry_date", "exit_date"],
    how="left",
    validate="one_to_one"
)

assert momentum_positions["effective_entry_price"].notna().all()
assert momentum_positions["effective_exit_price"].notna().all()

momentum_positions["shares"] = (
    momentum_positions["allocated_capital"]
    / (
        momentum_positions["effective_entry_price"]
        * (1 + COMMISSION_RATE)
    )
)

momentum_positions["exit_proceeds"] = (
    momentum_positions["shares"]
    * momentum_positions["effective_exit_price"]
    * (1 - COMMISSION_RATE)
)

daily_records = []
cash_balance = INITIAL_CAPITAL

for current_date in trading_dates:

    # Exit at the session open before processing entries.
    exiting = momentum_positions[
        momentum_positions["exit_date"] == current_date
    ]
    cash_balance += exiting["exit_proceeds"].sum()

    entering = momentum_positions[
        momentum_positions["entry_date"] == current_date
    ]
    cash_balance -= entering["allocated_capital"].sum()

    active = momentum_positions[
        (momentum_positions["entry_date"] <= current_date)
        & (momentum_positions["exit_date"] > current_date)
    ]

    market_value = 0.0

    for _, position in active.iterrows():
        closing_price = close_prices.loc[
            current_date, position["ticker"]
        ]
        assert pd.notna(closing_price)

        market_value += position["shares"] * closing_price

    daily_records.append({
        "date": current_date,
        "cash": cash_balance,
        "market_value": market_value,
        "equity": cash_balance + market_value,
        "open_positions": len(active)
    })

momentum_daily = pd.DataFrame(daily_records)

momentum_daily["daily_return"] = (
    momentum_daily["equity"].pct_change().fillna(0)
)

momentum_daily["cumulative_return"] = (
    momentum_daily["equity"] / INITIAL_CAPITAL - 1
)

momentum_daily["running_peak"] = (
    momentum_daily["equity"].cummax()
)

momentum_daily["drawdown"] = (
    momentum_daily["equity"]
    / momentum_daily["running_peak"] - 1
)

print("\nMOMENTUM DAILY EQUITY VALIDATION")
print("-" * 45)
print("Valuation rows:", len(momentum_daily))
print("First date:", momentum_daily["date"].min())
print("Last date:", momentum_daily["date"].max())
print("Final cash:", f"${momentum_daily['cash'].iloc[-1]:,.2f}")
print("Final market value:", f"${momentum_daily['market_value'].iloc[-1]:,.2f}")
print("Final equity:", f"${momentum_daily['equity'].iloc[-1]:,.2f}")
print("Remaining open positions:", momentum_daily["open_positions"].iloc[-1])

assert len(momentum_daily) == len(benchmark_daily)
assert momentum_daily["date"].equals(trading_dates)
assert momentum_daily["cash"].min() >= -1e-6
assert momentum_daily["open_positions"].max() <= MAX_POSITIONS
assert momentum_daily["equity"].notna().all()

print("Step 140A verification: PASSED")

# Step 140B - Momentum Risk-Adjusted Performance

valuation_days = (
    momentum_daily["date"].iloc[-1]
    - momentum_daily["date"].iloc[0]
).days

years = valuation_days / 365.25

daily_returns = momentum_daily["daily_return"].iloc[1:]

momentum_metrics = {
    "Cumulative Return (%)": (
        momentum_daily["equity"].iloc[-1]
        / INITIAL_CAPITAL - 1
    ) * 100,
    "CAGR (%)": (
        (
            momentum_daily["equity"].iloc[-1]
            / INITIAL_CAPITAL
        ) ** (1 / years) - 1
    ) * 100,
    "Annualized Volatility (%)": (
        daily_returns.std(ddof=1) * np.sqrt(252) * 100
    ),
    "Sharpe Ratio": (
        (daily_returns.mean() - RISK_FREE_RATE / 252)
        / daily_returns.std(ddof=1)
        * np.sqrt(252)
    ),
    "Maximum Drawdown (%)": (
        momentum_daily["drawdown"].min() * 100
    ),
    "Positive Days (%)": (
        (daily_returns > 0).mean() * 100
    ),
    "Return Periods": len(daily_returns)
}

print("\nMOMENTUM PERFORMANCE METRICS")
print("-" * 45)

for metric, value in momentum_metrics.items():
    if metric == "Return Periods":
        print(f"{metric}: {value}")
    else:
        print(f"{metric}: {value:.4f}")

assert years > 0
assert daily_returns.std(ddof=1) > 0
assert len(daily_returns) == len(momentum_daily) - 1
assert np.isfinite(
    list(momentum_metrics.values())
).all()

print("Step 140B verification: PASSED")

# Step 140C - Save Momentum Backtest Results

results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)

trades_output = results_dir / "momentum_executed_trades.csv"
equity_output = results_dir / "momentum_daily_equity.parquet"
metrics_output = results_dir / "momentum_performance_metrics.csv"

momentum_executed_df.to_csv(trades_output, index=False)
momentum_daily.to_parquet(equity_output, index=False)

pd.DataFrame(
    [
        {"metric": metric, "value": value}
        for metric, value in momentum_metrics.items()
    ]
).to_csv(metrics_output, index=False)

saved_trades = pd.read_csv(trades_output)
saved_equity = pd.read_parquet(equity_output)
saved_metrics = pd.read_csv(metrics_output)

assert len(saved_trades) == len(momentum_executed_df)
assert len(saved_equity) == len(momentum_daily)
assert len(saved_metrics) == len(momentum_metrics)
assert np.isclose(
    saved_equity["equity"].iloc[-1],
    momentum_daily["equity"].iloc[-1]
)

print("\nMOMENTUM OUTPUT FILES")
print("-" * 45)

for output in [trades_output, equity_output, metrics_output]:
    print(output.relative_to(PROJECT_ROOT))

print("Saved trades:", len(saved_trades))
print("Saved valuation dates:", len(saved_equity))
print("Saved metrics:", len(saved_metrics))
print("Step 140C verification: PASSED")
