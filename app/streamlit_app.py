# MarketPulse AI - Streamlit Dashboard

from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="MarketPulse AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


@st.cache_data
def load_data():
    trades = pd.read_parquet(
        RESULTS_DIR / "marketpulse_executed_trades.parquet"
    )

    portfolio = pd.read_parquet(
        RESULTS_DIR / "marketpulse_portfolio_daily.parquet"
    )

    benchmark = pd.read_parquet(
        RESULTS_DIR / "marketpulse_benchmark_daily.parquet"
    )

    performance = pd.read_csv(
        RESULTS_DIR / "marketpulse_performance_comparison.csv",
        index_col=0
    )

    return trades, portfolio, benchmark, performance


st.title("📈 MarketPulse AI")
st.caption("Stock Markets Analytics Zoomcamp 2026 | Capstone Project")

st.sidebar.title("MarketPulse AI")
st.sidebar.info("Stock Markets Analytics Zoomcamp 2026")

st.sidebar.markdown("### Dashboard Navigation")

st.sidebar.markdown("""
- [Overview](#data-verification)
- [Portfolio Performance](#portfolio-overview)
- [Trading Performance](#trading-performance)
- [Risk Analysis](#portfolio-risk-analysis)
- [Model Evaluation](#machine-learning-model-evaluation)
- [Strategy Insights](#strategy-insights)
- [Operational Monitoring](#operational-monitoring)
""")

st.sidebar.divider()

st.sidebar.caption(
    "Historical simulation | Educational research"
)

try:
    trades, portfolio, benchmark, performance = load_data()

    st.success("Backtest datasets loaded successfully.")

    st.subheader("Data Verification")

    col1, col2, col3 = st.columns(3)

    col1.metric("Executed Trades", f"{len(trades):,}")
    col2.metric("Trading Days", f"{len(portfolio):,}")
    col3.metric(
        "Final Portfolio Equity",
        f"${portfolio['equity'].iloc[-1]:,.2f}"
    )

    st.subheader("Performance Comparison")
    st.dataframe(performance, use_container_width=True)

    st.info(
        "Results are simulated and subject to backtesting "
        "and model reliability limitations."
    )

except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
    st.error(f"Unable to load dashboard data: {exc}")
    st.stop()

# --------------------------------------------------
# Portfolio Overview - Interactive Equity Curve
# --------------------------------------------------

import plotly.graph_objects as go

st.divider()
st.header("Portfolio Overview")

equity_chart = go.Figure()

equity_chart.add_trace(
    go.Scatter(
        x=benchmark["date"],
        y=benchmark["equity"],
        mode="lines",
        name="MarketPulse AI",
        line=dict(color="#2563EB", width=2.5)
    )
)

equity_chart.add_trace(
    go.Scatter(
        x=benchmark["date"],
        y=benchmark["spy_equity"],
        mode="lines",
        name="SPY Buy & Hold",
        line=dict(color="#F59E0B", width=2)
    )
)

equity_chart.update_layout(
    title="Portfolio Equity vs SPY Benchmark",
    xaxis_title="Date",
    yaxis_title="Portfolio Value ($)",
    hovermode="x unified",
    template="plotly_white",
    height=500,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

st.plotly_chart(
    equity_chart,
    use_container_width=True
)

st.caption(
    "Historical simulated performance. "
    "Past results do not guarantee future returns."
)

# --------------------------------------------------
# Trading Performance Analytics
# --------------------------------------------------

st.divider()
st.header("Trading Performance")

winning_trades = trades[trades["pnl"] > 0]
losing_trades = trades[trades["pnl"] < 0]

win_rate = len(winning_trades) / len(trades) * 100

gross_profit = winning_trades["pnl"].sum()
gross_loss = abs(losing_trades["pnl"].sum())

profit_factor = (
    gross_profit / gross_loss
    if gross_loss > 0
    else float("inf")
)

col1, col2, col3 = st.columns(3)

col1.metric("Win Rate", f"{win_rate:.2f}%")
col2.metric("Profit Factor", f"{profit_factor:.4f}")
col3.metric(
    "Net Trading P&L",
    f"${trades['pnl'].sum():,.2f}"
)

ticker_pnl = (
    trades.groupby("ticker", as_index=False)
    .agg(
        total_pnl=("pnl", "sum"),
        trade_count=("pnl", "size")
    )
    .sort_values("total_pnl", ascending=False)
)

import plotly.express as px

fig_ticker = px.bar(
    ticker_pnl,
    x="ticker",
    y="total_pnl",
    color="total_pnl",
    color_continuous_scale="RdYlGn",
    title="Net Trading Profit and Loss by Ticker",
    labels={
        "ticker": "Stock Ticker",
        "total_pnl": "Net P&L ($)"
    },
    hover_data=["trade_count"]
)

fig_ticker.update_layout(
    template="plotly_white",
    height=450,
    coloraxis_showscale=False
)

st.plotly_chart(
    fig_ticker,
    use_container_width=True
)

st.subheader("Executed Trade History")

selected_tickers = st.multiselect(
    "Filter by Stock Ticker",
    options=sorted(trades["ticker"].unique()),
    default=sorted(trades["ticker"].unique())
)

filtered_trades = trades[
    trades["ticker"].isin(selected_tickers)
]

st.dataframe(
    filtered_trades.sort_values(
        "entry_date",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Trade returns and P&L reflect the simulated "
    "backtest assumptions, including slippage."
)

# --------------------------------------------------
# Portfolio Risk Analysis
# --------------------------------------------------

st.divider()
st.header("Portfolio Risk Analysis")

strategy_max_dd = portfolio["drawdown"].min() * 100

spy_drawdown = (
    benchmark["spy_equity"]
    / benchmark["spy_equity"].cummax()
    - 1
)

spy_max_dd = spy_drawdown.min() * 100

risk_col1, risk_col2, risk_col3 = st.columns(3)

risk_col1.metric(
    "Strategy Maximum Drawdown",
    f"{strategy_max_dd:.2f}%"
)

risk_col2.metric(
    "SPY Maximum Drawdown",
    f"{spy_max_dd:.2f}%"
)

risk_col3.metric(
    "Average Open Positions",
    f"{portfolio['open_positions'].mean():.2f}"
)

# Drawdown comparison
drawdown_chart = go.Figure()

drawdown_chart.add_trace(
    go.Scatter(
        x=portfolio["date"],
        y=portfolio["drawdown"] * 100,
        name="MarketPulse AI",
        mode="lines",
        line=dict(color="#2563EB", width=2)
    )
)

drawdown_chart.add_trace(
    go.Scatter(
        x=benchmark["date"],
        y=spy_drawdown * 100,
        name="SPY Buy & Hold",
        mode="lines",
        line=dict(color="#F59E0B", width=2)
    )
)

drawdown_chart.update_layout(
    title="Portfolio Drawdown Comparison",
    xaxis_title="Date",
    yaxis_title="Drawdown (%)",
    hovermode="x unified",
    template="plotly_white",
    height=450
)

st.plotly_chart(
    drawdown_chart,
    use_container_width=True
)

# Open positions chart
positions_chart = px.area(
    portfolio,
    x="date",
    y="open_positions",
    title="Concurrent Open Positions",
    labels={
        "date": "Date",
        "open_positions": "Open Positions"
    }
)

positions_chart.update_layout(
    template="plotly_white",
    height=350,
    yaxis=dict(range=[0, 11])
)

st.plotly_chart(
    positions_chart,
    use_container_width=True
)

st.caption(
    "Risk metrics are based on historical simulated "
    "portfolio performance and may not represent "
    "future market conditions."
)

# --------------------------------------------------
# Machine Learning Model Evaluation
# --------------------------------------------------

st.divider()
st.header("Machine Learning Model Evaluation")

from sklearn.metrics import roc_auc_score, brier_score_loss

@st.cache_data
def load_model_predictions():
    return pd.read_parquet(
        RESULTS_DIR / "marketpulse_test_predictions.parquet"
    )

predictions = load_model_predictions()

actual = predictions["target_direction_10d"].astype(int)
probabilities = predictions["predicted_probability"]

model_auc = roc_auc_score(actual, probabilities)
model_brier = brier_score_loss(actual, probabilities)

base_rate = actual.mean()
baseline_brier = brier_score_loss(
    actual,
    [base_rate] * len(actual)
)

model_col1, model_col2, model_col3 = st.columns(3)

model_col1.metric(
    "Test ROC AUC",
    f"{model_auc:.4f}"
)

model_col2.metric(
    "Model Brier Score",
    f"{model_brier:.4f}"
)

model_col3.metric(
    "Constant Baseline Brier",
    f"{baseline_brier:.4f}"
)

st.warning(
    "The model shows weak out-of-sample discrimination. "
    "Its Brier score is worse than a retrospective "
    "constant-probability baseline."
)

# Probability calibration
calibration_data = predictions[
    ["target_direction_10d", "predicted_probability"]
].copy()

calibration_data["probability_bin"] = pd.cut(
    calibration_data["predicted_probability"],
    bins=np.linspace(0, 1, 11),
    include_lowest=True
)

calibration_plot = (
    calibration_data.groupby(
        "probability_bin",
        observed=False
    )
    .agg(
        mean_predicted=("predicted_probability", "mean"),
        observed_frequency=("target_direction_10d", "mean"),
        count=("target_direction_10d", "size")
    )
    .dropna()
    .reset_index()
)

calibration_chart = go.Figure()

calibration_chart.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name="Perfect Calibration",
        mode="lines",
        line=dict(color="gray", dash="dash")
    )
)

calibration_chart.add_trace(
    go.Scatter(
        x=calibration_plot["mean_predicted"],
        y=calibration_plot["observed_frequency"],
        name="Random Forest",
        mode="lines+markers",
        marker=dict(size=9),
        customdata=calibration_plot[["count"]],
        hovertemplate=(
            "Mean predicted: %{x:.2%}<br>"
            "Observed frequency: %{y:.2%}<br>"
            "Observations: %{customdata[0]}<extra></extra>"
        )
    )
)

calibration_chart.update_layout(
    title="Out-of-Sample Probability Calibration",
    xaxis_title="Mean Predicted Probability",
    yaxis_title="Observed Positive Frequency",
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1]),
    template="plotly_white",
    height=450
)

st.plotly_chart(
    calibration_chart,
    use_container_width=True
)

st.caption(
    "The constant baseline uses the test-period positive rate "
    "as a retrospective reference, not a deployable forecast. "
    "Calibration bins with few observations are unstable. "
    "Overlapping 10-day prediction horizons are not independent."
)


# --------------------------------------------------
# Strategy Insights and Limitations
# --------------------------------------------------

st.divider()
st.header("Strategy Insights")

analysis_dir = RESULTS_DIR / "strategy_analysis"

@st.cache_data
def load_strategy_insights():
    concentration = pd.read_csv(
        analysis_dir / "profit_concentration.csv"
    )
    robustness = pd.read_csv(
        analysis_dir / "robustness_summary.csv"
    )
    return concentration, robustness


concentration, robustness = load_strategy_insights()

st.subheader("Profit Concentration")

st.dataframe(
    concentration,
    use_container_width=True,
    hide_index=True
)

st.info(
    "The 10 largest winning trades contributed approximately "
    "79.30% of total net trading profit. This indicates "
    "substantial dependence on a small number of trades."
)

st.subheader("Strategy Robustness Summary")

st.dataframe(
    robustness,
    use_container_width=True,
    hide_index=True
)

st.subheader("Key Findings")

st.markdown("""
- **Simulated performance:** MarketPulse AI returned 25.33%,
  compared with 22.58% for SPY.
- **Risk:** The strategy experienced a smaller historical
  maximum drawdown than SPY.
- **Trading:** The strategy achieved a 51.84% win rate
  and a profit factor of 1.4213.
- **Concentration:** The largest winning trades accounted
  for a substantial share of net profit.
- **Model reliability:** Test ROC AUC was only 0.5125,
  and probability calibration was weaker than the
  retrospective constant baseline.
""")

st.subheader("Methodology Limitations")

with st.expander("View Backtesting and Model Limitations"):
    st.markdown("""
1. Adjusted OHLC prices were used as execution-price proxies.
2. SPY benchmark transaction costs were not aligned.
3. Fractional shares and fixed allocations were assumed.
4. The stock universe may introduce survivorship bias.
5. Macroeconomic data may have point-in-time availability
   and revision limitations.
6. Market impact and liquidity constraints were not modeled.
7. The evaluation used a single historical test period.
8. Overlapping prediction horizons are not independent.
9. Trading profits were concentrated in large winners.
""")

st.subheader("Final Assessment")

st.warning(
    "MarketPulse AI demonstrated positive simulated returns, "
    "but the weak predictive model and backtesting limitations "
    "do not establish a robust or deployable trading advantage."
)

st.caption(
    "MarketPulse AI | Stock Markets Analytics Zoomcamp 2026 "
    "| Educational research project — not investment advice."
)



# --------------------------------------------------
# Operational Monitoring - Local Pipeline Execution
# --------------------------------------------------

st.divider()
st.header("Operational Monitoring")


st.subheader("Data Freshness Monitoring")

data_sources = {
    "Stock Prices": "stock_prices.parquet",
    "FRED Macroeconomic": "fred_macro.parquet",
    "Market Sentiment": "market_sentiment.parquet",
}

freshness_records = []

for source_name, filename in data_sources.items():
    data_path = PROJECT_ROOT / "data" / filename

    if data_path.exists():
        try:
            source_data = pd.read_parquet(data_path)

            latest_date = (
                pd.to_datetime(source_data["date"]).max()
                if "date" in source_data.columns
                else pd.NaT
            )

            modified_at = pd.Timestamp(
                data_path.stat().st_mtime, unit="s", tz="UTC"
            ).tz_convert("America/Chicago")

            freshness_records.append({
                "Dataset": source_name,
                "Records": len(source_data),
                "Latest Observation": (
                    latest_date.strftime("%Y-%m-%d")
                    if pd.notna(latest_date) else "Unknown"
                ),
                "File Updated (Central)": modified_at.strftime(
                    "%Y-%m-%d %I:%M:%S %p %Z"
                ),
            })
        except (OSError, ValueError, KeyError) as exc:
            st.warning(f"Unable to inspect {source_name}: {exc}")
    else:
        freshness_records.append({
            "Dataset": source_name,
            "Records": None,
            "Latest Observation": "Unavailable",
            "File Updated (Central)": "Unavailable",
        })

st.dataframe(
    pd.DataFrame(freshness_records),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Latest observation dates indicate dataset coverage, not "
    "necessarily the publication date of each underlying indicator. "
    "Local data files are not automatically available in Streamlit Cloud."
)


st.caption(
    "Pipeline execution history | Local monitoring"
)

history_path = PROJECT_ROOT / "logs" / "pipeline_execution_history.jsonl"

if history_path.exists():
    try:
        history = pd.read_json(history_path, lines=True, encoding="utf-8-sig")

        if history.empty:
            st.info("No pipeline execution records are available.")
        else:
            history["started_at"] = pd.to_datetime(
                history["started_at"], utc=True
            )
            history["completed_at"] = pd.to_datetime(
                history["completed_at"], utc=True
            )

            history = history.sort_values(
                "started_at", ascending=False
            )

            for column in ("started_at", "completed_at"):
                history[column] = (
                    history[column]
                    .dt.tz_convert("America/Chicago")
                    .dt.strftime("%Y-%m-%d %I:%M:%S %p %Z")
                )

            st.warning(
                "Current execution history includes simulated "
                "monitoring tests. These records do not verify "
                "a successful production pipeline run."
            )

            production_history = history[
                history["execution_type"] == "production"
            ]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Production Executions", len(production_history)
            )
            col2.metric(
                "Production Successes",
                int((production_history["status"] == "success").sum())
            )
            col3.metric(
                "Production Failures",
                int((production_history["status"] == "failed").sum())
            )

            if production_history.empty:
                st.info(
                    "No production pipeline executions have been "
                    "recorded yet. The history below contains "
                    "monitoring test records."
                )

            st.subheader("Pipeline Execution History")

            st.dataframe(
                history[
                    [
                        "started_at",
                        "completed_at",
                        "duration_seconds",
                        "status",
                        "execution_type",
                        "exit_code",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    except (ValueError, KeyError, OSError) as exc:
        st.error(f"Unable to load monitoring history: {exc}")
else:
    st.info(
        "Local pipeline execution history is not available "
        "in this environment. Streamlit Cloud does not "
        "automatically synchronize Windows pipeline logs."
    )
