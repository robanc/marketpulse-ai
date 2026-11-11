# MarketPulse AI
**Stock Markets Analytics Zoomcamp 2026 — Capstone Project**

## Table of Contents

1. [Problem Description](#1-problem-description)
2. [Data Sources](#2-data-sources)
3. [Data Transformations + EDA](#3-data-transformations--eda)
4. [Modeling](#4-modeling)
5. [Trading Simulation](#5-trading-simulation)
6. [Automation](#6-automation)
7. [Reproducibility](#7-reproducibility)
8. [Advanced Features and Enhancements](#8-advanced-features-and-enhancements)
9. [Results and Key Findings](#9-results-and-key-findings)
10. [Limitations and Responsible Use](#10-limitations-and-responsible-use)
11. [References and Acknowledgments](#11-references-and-acknowledgments)



---

# 1. Problem Description

## 1.1 Background and Problem Statement

Financial markets are influenced by multiple factors, including historical stock prices, trading volume, macroeconomic conditions, market sentiment, and investor behavior.

Traditional trading strategies often rely on individual technical indicators or simple price-based rules. However, these approaches may overlook relationships between broader economic conditions, market sentiment, and individual stock performance.

**MarketPulse AI** addresses this problem by developing a machine learning-powered stock market analytics and trading simulation system that combines multiple data sources to generate predictive signals and evaluate systematic investment strategies.

The project investigates whether combining technical indicators, macroeconomic information, and market sentiment can produce useful predictions of future stock returns and support risk-aware portfolio decisions.

## 1.2 Project Objectives and End Goal

MarketPulse AI aims to:

1. Collect historical equity market data, macroeconomic indicators, and market sentiment information.
2. Integrate the datasets into a unified analytical framework.
3. Engineer predictive features representing price behavior, momentum, volatility, market conditions, and sentiment.
4. Train and compare machine learning models for stock return and directional prediction.
5. Generate trading signals using model predictions and probability-based decision rules.
6. Simulate historical portfolio performance with capital allocation and risk management constraints.
7. Compare trading results against the SPY S&P 500 ETF benchmark.
8. Analyze profitability, drawdowns, model quality, and strategy robustness.
9. Present results through an interactive Streamlit dashboard.

**End Goal:** Develop a reproducible research framework that connects market data, machine learning predictions, trading simulation, and portfolio performance analysis to help evaluate systematic trading ideas.

The system is designed for research and educational purposes rather than live investment execution.

## 1.3 What Makes MarketPulse AI Different?

Unlike a basic weekly trading strategy based only on the largest stocks, MarketPulse AI uses a broader analytical framework.

**Multi-asset analysis:** The project evaluates 20 stocks across multiple sectors, using SPY as the market benchmark.

**Longer prediction horizon:** Instead of focusing exclusively on short-term weekly trading, the project predicts returns over a 10-trading-day forward horizon.

**Multiple data sources:** Historical stock prices are combined with Federal Reserve macroeconomic indicators and the CNN Fear & Greed Index.

**Feature-rich modeling:** The project develops 33 engineered features to capture technical behavior, market conditions, and sentiment.

**Model comparison:** Decision Tree, Random Forest, and XGBoost models are evaluated, alongside baseline approaches.

**Portfolio-level evaluation:** Predictions are assessed not only through machine learning metrics but also through simulated trading performance, risk exposure, and benchmark comparison.

**Interactive analytics:** A Streamlit dashboard presents model evaluation, portfolio performance, trading activity, risk analysis, and strategy insights.

## 1.4 End-to-End Workflow

MarketPulse AI follows five analytical stages.

| Stage | Notebook | Purpose |
|---|---|---|
| 1. Data Collection | `01_data_collection.ipynb` | Download, clean, align, and integrate market, macroeconomic, and sentiment data |
| 2. EDA and Feature Engineering | `02_eda_feature_engineering.ipynb` | Analyze variables, engineer features, define targets, and create chronological data splits |
| 3. Model Training | `03_model_training.ipynb` | Train, tune, compare, and evaluate predictive models |
| 4. Trading Backtest | `04_trading_backtest.ipynb` | Convert predictions into trading decisions and simulate historical portfolio performance |
| 5. Strategy Analysis | `05_strategy_analysis.ipynb` | Examine profitability, drawdowns, ticker performance, model calibration, and robustness |

The outputs are presented through `app/streamlit_app.py`.

## 1.5 Initial Findings

The historical backtest produced the following results:

| Metric | MarketPulse AI |
|---|---:|
| Simulated starting capital | $100,000 |
| Simulated ending capital | $125,326.20 |
| Strategy cumulative return | +25.33% |
| SPY benchmark cumulative return | +22.58% |
| Executed trades | 299 |
| Strategy maximum drawdown | -5.24% |
| SPY maximum drawdown | -8.88% |
| Selected model test ROC AUC | 0.5125 |

The strategy outperformed the SPY benchmark in cumulative return over the evaluated historical period. However, the selected model's ROC AUC of 0.5125 indicates weak directional predictive performance.

These findings highlight an important distinction between historical simulated profitability and reliable predictive capability.

The backtest is subject to execution assumptions, data-availability limitations, and potential sources of bias. The results should not be interpreted as evidence of future profitability.

Further details are provided in the Modeling, Trading Simulation, and Strategy Analysis sections.


---

# 2. Data Sources

MarketPulse AI combines three complementary data sources to evaluate stock market behavior, macroeconomic conditions, and investor sentiment.

## 2.1 Yahoo Finance — Equity Market Data

Historical daily equity data is collected using the `yfinance` Python library.

The dataset covers 20 U.S. stocks across multiple sectors, with SPY serving as the S&P 500 benchmark.

**Stocks:** AAPL, MSFT, NVDA, GOOGL, AMZN, JPM, BAC, GS, JNJ, UNH, PFE, WMT, COST, HD, XOM, CVX, CAT, GE, META, NFLX.

**Benchmark:** SPY.

The market dataset contains daily open, high, low, close, and trading volume observations.

These variables support the calculation of momentum, moving averages, volatility, technical indicators, and volume-based features.

## 2.2 FRED — Macroeconomic Indicators

Macroeconomic data is obtained from the Federal Reserve Economic Data (FRED) platform.

| FRED Series | Variable | Description |
|---|---|---|
| DGS10 | `treasury_10y` | 10-year Treasury yield |
| DGS2 | `treasury_2y` | 2-year Treasury yield |
| FEDFUNDS | `fed_funds_rate` | Effective federal funds rate |
| CPIAUCSL | `cpi` | Consumer Price Index |
| UNRATE | `unemployment_rate` | U.S. unemployment rate |

These indicators provide information about interest rates, inflation, monetary policy, and economic conditions.

## 2.3 CNN Fear & Greed Index — Alternative Sentiment Data

MarketPulse AI introduces the CNN Fear & Greed Index as an additional data source beyond Yahoo Finance and FRED.

The index summarizes broad investor sentiment on a scale from 0 to 100, where lower values indicate greater fear and higher values indicate greater greed.

The dataset includes:

- `fear_greed_index` — Numerical market sentiment score.
- `value_classification` — Sentiment category.

This additional source allows MarketPulse AI to incorporate sentiment-based features alongside technical and macroeconomic indicators.

## 2.4 Dataset Coverage and Integration

The historical market dataset covers January 2018 through November 2026.

After aligning the market, macroeconomic, and sentiment datasets, the integrated dataset contains approximately 46,221 stock-date observations across 21 instruments, including SPY.

Data is aligned by trading date and ticker, with market-wide indicators associated with the corresponding daily stock observations.

The complete engineered dataset is further filtered to retain observations with available predictive features and known targets for model development.

The dataset contains fewer than one million observations; therefore, MarketPulse AI does not claim to satisfy the evaluation criterion for datasets exceeding one million records.

## 2.5 Engineered Feature Dictionary

MarketPulse AI generates 33 engineered features grouped into seven categories.

### A. Price Momentum — 4 Features

| Feature | Description |
|---|---|
| `return_1d` | One-trading-day stock return |
| `return_5d` | Five-trading-day stock return |
| `return_10d` | Ten-trading-day stock return |
| `return_20d` | Twenty-trading-day stock return |

### B. Moving Averages — 6 Features

| Feature | Description |
|---|---|
| `sma_5` | Five-day simple moving average |
| `sma_10` | Ten-day simple moving average |
| `sma_20` | Twenty-day simple moving average |
| `sma_50` | Fifty-day simple moving average |
| `ema_12` | Twelve-day exponential moving average |
| `ema_26` | Twenty-six-day exponential moving average |

### C. Volatility — 4 Features

| Feature | Description |
|---|---|
| `volatility_5d` | Five-day rolling return volatility |
| `volatility_20d` | Twenty-day rolling return volatility |
| `daily_range_pct` | Daily high-to-low price range expressed as a percentage |
| `atr_14` | Fourteen-day Average True Range |

### D. Technical Indicators — 6 Features

| Feature | Description |
|---|---|
| `rsi_14` | Fourteen-day Relative Strength Index |
| `macd` | Moving Average Convergence Divergence |
| `macd_signal` | MACD signal line |
| `macd_histogram` | Difference between MACD and its signal line |
| `bollinger_upper` | Upper Bollinger Band |
| `bollinger_lower` | Lower Bollinger Band |

### E. Trading Volume — 4 Features

| Feature | Description |
|---|---|
| `volume_change_1d` | One-day percentage change in trading volume |
| `volume_sma_20` | Twenty-day average trading volume |
| `relative_volume_20` | Trading volume relative to its twenty-day average |
| `log_volume` | Logarithmic transformation of trading volume |

### F. Macroeconomic Indicators — 5 Features

| Feature | Description |
|---|---|
| `yield_spread` | Difference between 10-year and 2-year Treasury yields |
| `treasury_10y_change_5d` | Five-trading-day change in the 10-year Treasury yield |
| `treasury_2y_change_5d` | Five-trading-day change in the 2-year Treasury yield |
| `fed_funds_change_20d` | Twenty-trading-day change in the federal funds rate |
| `cpi_change_60d` | Sixty-trading-day percentage change in CPI |

### G. Market Sentiment — 4 Features

| Feature | Description |
|---|---|
| `sentiment_change_1d` | One-day change in the Fear & Greed Index |
| `sentiment_sma_5` | Five-day average sentiment score |
| `sentiment_sma_20` | Twenty-day average sentiment score |
| `sentiment_volatility_20` | Twenty-day standard deviation of sentiment |

## 2.6 Prediction Targets

Two prediction targets are generated:

| Target | Description |
|---|---|
| `target_return_10d` | Forward 10-trading-day stock return |
| `target_direction_10d` | Binary indicator of whether the forward return is positive |

The prediction horizon is ten trading days.

The targets are calculated using future stock prices, while predictive features are derived from historical observations.

The future-price helper variable is excluded from the training dataset to prevent direct target leakage.

## 2.7 Data Storage

Processed datasets are stored locally in Parquet format for efficient reuse.

Important output files include:

- `data/stock_prices.parquet`
- `data/fred_macro.parquet`
- `data/market_sentiment.parquet`
- `data/marketpulse_merged.parquet`
- `data/marketpulse_features.parquet`
- `data/marketpulse_training.parquet`

The `data/` directory is excluded from Git version control. Users must execute the data collection and feature engineering notebooks to regenerate these datasets.


---

# 3. Data Transformations + EDA

## 3.1 Data Integration and Cleaning

MarketPulse AI integrates three datasets into a unified analytical dataset:

- **Yahoo Finance:** Daily stock prices and trading volume.
- **FRED:** Macroeconomic indicators, including Treasury yields, interest rates, inflation, and unemployment.
- **CNN Fear & Greed Index:** Daily market sentiment scores.

The datasets are aligned by trading date, with macroeconomic and sentiment observations associated with individual stock records.

The resulting integrated dataset contains approximately 46,221 observations across 21 instruments, including SPY.

The data preparation workflow includes date alignment, chronological sorting, duplicate checks, missing-value analysis, and validation of engineered features.

The integrated dataset is stored in `data/marketpulse_merged.parquet`.

## 3.2 Exploratory Data Analysis

Notebook `02_eda_feature_engineering.ipynb` performs exploratory analysis before model development.

The analysis includes:

| Analysis | Purpose |
|---|---|
| Dataset overview and descriptive statistics | Examine distributions, coverage, and data quality |
| Normalized stock prices | Compare historical price movements across stocks |
| Daily returns and volatility | Investigate return behavior and risk |
| Stock return correlations | Identify relationships between stock returns |
| Maximum drawdown | Evaluate historical downside risk |
| Market sentiment and SPY returns | Explore relationships between investor sentiment and market performance |
| Macroeconomic indicators and SPY returns | Examine market behavior under different economic conditions |

These analyses provide context for selecting predictive features and understanding the characteristics of the historical dataset.

The notebook examines correlations between stock returns. A dedicated correlation analysis evaluates all 33 engineered features against the 10-trading-day forward return target.

**Feature-to-Target Correlation Analysis**

An additional Pearson correlation analysis was performed between all 33 engineered features and the 10-trading-day forward return target (`target_return_10d`), excluding SPY benchmark observations.

The strongest positive correlations were `volatility_20d` (+0.0765), `volatility_5d` (+0.0576), and `daily_range_pct` (+0.0482).

The strongest negative correlations were `cpi_change_60d` (−0.0900), `fed_funds_change_20d` (−0.0751), and `macd_signal` (−0.0424).

All correlations were relatively weak, suggesting limited individual linear relationships with future returns. These results are descriptive and should not be interpreted as evidence of causal relationships or out-of-sample predictive performance.

The complete results are saved in `results/feature_target_correlations.csv` and generated by Notebook 02 and the exported EDA pipeline.

## 3.3 Feature Engineering

MarketPulse AI generates **33 engineered features** across seven categories.

| Feature category | Count | Examples |
|---|---:|---|
| Price momentum | 4 | `return_5d`, `return_20d` |
| Moving averages | 6 | `sma_20`, `ema_12` |
| Volatility | 4 | `volatility_20d`, `atr_14` |
| Technical indicators | 6 | `rsi_14`, `macd` |
| Trading volume | 4 | `relative_volume_20`, `log_volume` |
| Macroeconomic indicators | 5 | `yield_spread`, `cpi_change_60d` |
| Market sentiment | 4 | `sentiment_sma_20`, `sentiment_volatility_20` |
| **Total** | **33** | |

These transformations capture multiple dimensions of market behavior, including historical returns, trend strength, volatility, trading activity, interest-rate changes, inflation, and investor sentiment.

The complete feature dictionary is provided in Section 2.5.

## 3.4 Prediction Targets and Feature Sets

The project uses a 10-trading-day prediction horizon.

Two target variables are created:

| Variable | Description |
|---|---|
| `target_return_10d` | Percentage return over the next ten trading days |
| `target_direction_10d` | Binary classification of whether the future return is positive |

The 33 engineered features are explicitly identified in the notebook through the `feature_columns` list.

The targets are separately identified through `target_columns`.

The project also separates stock observations from SPY benchmark observations. SPY is excluded from the training dataset and reserved for market comparison.

This prevents the benchmark instrument from being treated as an ordinary stock training observation.

## 3.5 Data Quality and Model Preparation

After feature engineering, the notebook performs quality checks to identify:

- Duplicate ticker-date records.
- Missing engineered feature values.
- Infinite feature values.
- Rows with complete predictive features.
- Rows with both complete features and known targets.

Only observations with complete engineered features and available targets are retained for the model-ready dataset.

The future-price helper variable, `future_close_10d`, is removed from the training dataset to prevent direct leakage of future prices into the predictive features.

The resulting datasets are saved as:

- `data/marketpulse_features.parquet` — Complete engineered dataset.
- `data/marketpulse_training.parquet` — Model-ready stock observations.

## 3.6 Chronological Data Splitting and Leakage Prevention

MarketPulse AI uses chronological training, validation, and testing periods rather than random splitting.

The modeling workflow applies an approximately 70% training, 15% validation, and 15% testing split, with a ten-trading-day purge between periods to reduce overlap in forward-return targets.

This approach is more appropriate for financial time-series modeling because it helps preserve the historical ordering of observations.

Additional considerations remain important, including the publication timing and historical revisions of macroeconomic indicators, as well as the timing of sentiment observations.

Consequently, the split and removal of future-price variables reduce specific forms of look-ahead leakage but do not establish that the entire dataset is point-in-time accurate.

## 3.7 Key Findings and Implications

The exploratory analysis and transformation workflow establish a consistent foundation for predictive modeling.

Key outcomes include:

1. Integration of equity market, macroeconomic, and sentiment information into one analytical dataset.
2. Development of 33 predictive features, substantially exceeding the rubric's minimum of five transformed features.
3. Examination of historical stock performance, volatility, drawdowns, and return correlations.
4. Creation of two forward-looking prediction targets.
5. Preparation of a labeled training dataset with complete engineered features.
6. Separation of SPY benchmark observations from model training.
7. Chronological splitting to support out-of-sample evaluation.

These outputs provide the inputs for the machine learning models described in Section 4.


---

# 4. Modeling

## 4.1 Modeling Objectives and Baseline Evaluation

MarketPulse AI investigates whether machine learning models can predict the direction of individual stock returns over the next ten trading days.

The primary classification target is `target_direction_10d`, indicating whether the forward return is positive.

The modeling workflow begins with baseline evaluation to establish a reference for comparing machine learning performance.

Models are evaluated using classification metrics such as accuracy, precision, recall, F1 score, and ROC AUC.

## 4.2 Machine Learning Model Comparison

Three machine learning algorithms are implemented and evaluated:

| Model | Purpose |
|---|---|
| Decision Tree Classifier | Establish an interpretable tree-based predictive model |
| Random Forest Classifier | Capture nonlinear relationships using an ensemble of decision trees |
| XGBoost Classifier | Evaluate a gradient-boosted tree model as an advanced alternative |

The models are trained using historical observations and evaluated using chronologically separated validation data.

This comparison supports model selection based on out-of-sample validation performance rather than training accuracy alone.

## 4.3 Historical Feature Availability Audit

The modeling workflow includes an audit of historical feature availability.

This is important because some financial indicators may not have been available to investors at the timestamps assigned to historical records.

The project subsequently evaluates a market-only feature subset containing 24 predictive variables, excluding the macroeconomic and sentiment-derived features from the final classifier.

This experiment helps reduce dependence on potentially delayed or retrospectively revised external indicators.

The original 33-feature dataset remains available for exploratory analysis and model comparison.

## 4.4 Hyperparameter Tuning

The selected market-only Random Forest is tuned using six combinations of maximum tree depth and minimum samples per leaf.

| Maximum depth | Minimum samples per leaf |
|---:|---:|
| 5 | 50 |
| 10 | 50 |
| 15 | 50 |
| 10 | 100 |
| 15 | 100 |
| 20 | 100 |

The experiments use 100 trees, `max_features=0.7`, and `random_state=42`.

The best configuration is selected according to validation ROC AUC.

**Selected hyperparameters:**

| Parameter | Value |
|---|---|
| Model | Random Forest Classifier |
| Number of trees | 100 |
| Maximum depth | 15 |
| Minimum samples per leaf | 100 |
| Maximum features | 0.7 |
| Random state | 42 |
| Number of predictive features | 24 |

The selected model achieved a validation ROC AUC of approximately 0.5536.

## 4.5 Probability-Based Trading Decision Rules

Rather than relying exclusively on binary predictions, MarketPulse AI evaluates the probability that a stock will generate a positive return over the next ten trading days.

The validation workflow compares probability thresholds of:

- 0.50
- 0.55
- 0.60
- 0.65
- 0.70

For each threshold, the project evaluates the number of selected observations, selection rate, precision, recall, and F1 score.

The workflow also examines the historical forward returns of observations selected at each threshold.

The final trading signal rule is:

**Generate a positive trading signal when the predicted probability of a positive ten-day return is at least 0.60.**

This threshold is selected using validation-period analysis and locked before final test evaluation.

## 4.6 Final Model Selection and Feature Importance

The selected classifier is a market-only Random Forest trained using 24 predictive features.

Feature importance is calculated from the fitted model, and the 15 most important features are visualized.

This analysis helps identify which market-derived variables contribute most strongly to the model's predictions.

Random Forest feature importance reflects the fitted model's internal structure and should not be interpreted as proof of causal relationships.

## 4.7 Out-of-Sample Test Evaluation

After selecting the model and locking the probability threshold, MarketPulse AI evaluates the final configuration on the held-out test dataset.

The test workflow calculates:

- Accuracy
- Precision
- Recall
- F1 score
- ROC AUC
- Confusion matrix
- Trading signal selection rate
- Mean and median forward returns of selected observations
- Positive outcome rate

The final model achieved:

| Metric | Result |
|---|---:|
| Validation ROC AUC | 0.5536 |
| Test ROC AUC | 0.5125 |
| Probability threshold | 0.60 |
| Test observations | 6,400 |

The decrease in ROC AUC between validation and test periods indicates that the model's ability to distinguish positive from negative future returns did not generalize strongly to the held-out period.

The test ROC AUC of 0.5125 is only slightly above the 0.50 reference associated with random ranking.

Consequently, the predictive model should be considered experimental rather than a reliable forecasting system.

## 4.8 Modeling Findings and Limitations

The modeling experiments demonstrate several important findings:

1. Multiple machine learning approaches can be evaluated within a consistent chronological framework.
2. Hyperparameter tuning provides a systematic method for selecting model configurations.
3. Historical feature availability matters when working with macroeconomic and sentiment information.
4. Probability thresholds provide a flexible mechanism for controlling trading signal selection.
5. Validation performance does not necessarily translate into strong performance on unseen market periods.
6. Predictive classification metrics and simulated trading profitability must be evaluated separately.

Future improvements may include walk-forward validation, probability calibration, alternative target definitions, regime-specific models, and more robust feature availability controls.

The trading simulation and financial performance of the selected model are discussed in Section 5.


---

# 5. Trading Simulation

## 5.1 Trading Strategy and Objectives

MarketPulse AI evaluates whether machine learning-generated trading signals can support a systematic, risk-constrained stock portfolio.

The strategy uses the selected Random Forest classifier to estimate the probability that an individual stock will generate a positive return over the next ten trading days.

A trading candidate is generated when the predicted probability is at least 0.60.

The strategy uses a fixed ten-trading-session holding period and evaluates its historical performance against an SPY buy-and-hold benchmark.

## 5.2 Signal Generation and Trade Execution

Trading signals are generated using the locked model configuration established during validation.

The simulation follows these rules:

| Parameter | Configuration |
|---|---|
| Model | Market-only Random Forest |
| Prediction horizon | 10 trading days |
| Probability threshold | 0.60 |
| Signal timing | After market close |
| Entry timing | Next trading session |
| Holding period | 10 trading sessions |
| Trade direction | Long-only |
| Candidate priority | Higher predicted probability first |

Eligible candidates are ordered by entry date, predicted probability, and ticker.

The execution engine processes candidates chronologically while enforcing capital availability and position constraints.

## 5.3 Portfolio Capital Allocation

The simulation begins with $100,000 in capital.

Each executed position receives a fixed allocation of $10,000, subject to available cash.

The portfolio permits a maximum of ten simultaneous positions.

The execution engine:

1. Releases capital from positions that have reached their exit date.
2. Rejects a new trade if the same ticker is already held.
3. Rejects a trade if the maximum number of positions has been reached.
4. Rejects a trade if insufficient cash is available.
5. Allocates capital to eligible positions.
6. Returns exit proceeds to available cash when positions close.

This design prevents the simulation from allocating more capital than is available.

Realized gains and losses affect subsequent cash availability, although the nominal allocation remains fixed at $10,000 per position.

## 5.4 Transaction Costs and Execution Assumptions

The backtest incorporates transaction-cost adjustments when calculating net trade returns.

It uses adjusted OHLC prices as an execution-price proxy and assumes fractional shares are permitted.

The daily equity calculation tracks:

- Available cash.
- Market value of open positions.
- Total portfolio equity.
- Number of open positions.
- Daily portfolio returns.
- Cumulative returns.
- Portfolio drawdowns.

The simulation does not fully reproduce live brokerage execution.

In particular, it does not model market impact or liquidity constraints, and the SPY benchmark's transaction-cost treatment is not fully aligned with the strategy.

These limitations are important when interpreting historical profitability.

## 5.5 Corrected Portfolio Execution Engine

Notebook 04 includes preliminary calculations followed by a corrected portfolio execution engine.

The corrected implementation processes all valid trading candidates in chronological order and applies the portfolio constraints directly during trade selection.

It also reconciles realized trade profit and loss against the final portfolio equity.

The final performance results reported in this project are based on the corrected execution engine, not the preliminary simulation.

The notebook verifies:

- No overlapping positions in the same ticker.
- No more than ten concurrent positions.
- No negative cash balances.
- Correct ten-session holding periods.
- No executed trades below the probability threshold.
- Trade entry dates occurring after signal dates.
- Reconciliation between trade-level P&L and portfolio equity.

## 5.6 Trading Performance and SPY Benchmark

The corrected historical backtest produced the following results:

| Metric | MarketPulse AI | SPY Buy-and-Hold |
|---|---:|---:|
| Initial capital | $100,000 | $100,000 |
| Final capital | $125,326.20 | Approximately $122,580 |
| Cumulative return | +25.33% | +22.58% |
| Maximum drawdown | -5.24% | -8.88% |

The strategy generated 299 executed trades.

Over the evaluated period, MarketPulse AI exceeded SPY's cumulative return by approximately 2.75 percentage points.

The strategy also experienced a smaller historical maximum drawdown.

These results represent simulated performance under the assumptions described above.

They do not establish that the strategy will outperform SPY in future periods.

## 5.7 Risk-Adjusted Performance Evaluation

The notebook calculates additional portfolio performance metrics, including:

- Cumulative return.
- Annualized return.
- Annualized volatility.
- Sharpe ratio.
- Maximum drawdown.
- Positive daily return rate.
- Number of trading days.

Annualized return and volatility are calculated using a 252-trading-day convention.

The Sharpe ratio calculation assumes a zero risk-free rate.

These measures support comparison between the strategy and SPY beyond cumulative returns alone.

## 5.8 Strategy Analysis and Robustness

Notebook `05_strategy_analysis.ipynb` extends the backtest through deeper exploratory analysis.

The analysis examines:

- Individual trade performance.
- Performance differences across stock tickers.
- Portfolio concentration.
- Capital exposure and utilization.
- Drawdown characteristics.
- Prediction probability calibration.
- Strategy robustness and potential improvements.

These analyses help identify where the strategy performs well, where it struggles, and whether apparent profitability is concentrated in particular stocks or market conditions.

The findings provide a foundation for future improvements to model selection, signal filtering, portfolio allocation, and risk management.

## 5.9 Backtest Integrity and Limitations

The backtest includes explicit integrity checks, but several methodological limitations remain:

1. Adjusted OHLC prices are used as execution-price proxies.
2. Fractional shares are assumed.
3. Position sizes are fixed at $10,000.
4. The benchmark does not use fully aligned transaction-cost assumptions.
5. Current stock constituent selection may introduce survivorship bias.
6. Market impact and liquidity constraints are not modeled.
7. The selected classifier's test ROC AUC of 0.5125 indicates weak directional predictive performance.

The simulation demonstrates a functioning historical research framework, but its profitability should not be interpreted as evidence of a dependable live trading advantage.

## 5.10 Trading Simulation Conclusions

MarketPulse AI demonstrates how probability-based machine learning signals can be incorporated into a constrained portfolio execution framework.

The corrected simulation includes realistic elements such as available-cash enforcement, concurrent position limits, trade overlap prevention, transaction-cost adjustments, and daily portfolio valuation.

The historical strategy achieved a cumulative return of 25.33%, compared with 22.58% for SPY, while recording a smaller maximum drawdown.

However, the modest outperformance, weak out-of-sample model discrimination, and unresolved execution assumptions require further investigation.

Future work should evaluate additional strategies, more realistic execution prices, transaction-cost consistency, walk-forward testing, and robustness across different market regimes.


---

## 5.11 Alternative Momentum Strategy and Three-Strategy Comparison

To evaluate an independent, rule-based alternative to the Random Forest
classifier, MarketPulse AI implements a ten-day positive-momentum strategy
in `src/pipeline/06_momentum_backtest.py`.

### Momentum signal and execution rules

- Universe: the same 20 equities used in the Random Forest backtest; SPY
  remains the market benchmark.
- Signal: ten-session historical closing-price return greater than zero.
- Ranking: eligible stocks are prioritized by descending ten-session momentum.
- Entry: next trading session's opening price after signal generation.
- Exit: opening price ten trading sessions after entry.
- Initial capital: $100,000.
- Position allocation: fixed $10,000 per trade, with fractional shares.
- Portfolio capacity: at most ten simultaneous positions.
- Risk controls: no overlapping positions in the same ticker, no entries
  exceeding available cash, and explicit exit-before-entry processing.
- Transaction costs: 0.05% slippage per side and zero commission.
- Risk-free rate for Sharpe calculations: zero.

The momentum strategy does not use Random Forest probabilities for its
trading decisions. Both active strategies are evaluated on the same
historical universe and valuation dates.

### Momentum execution results

The strategy generated 3,478 valid trade candidates and executed 320
trades. The remaining candidates were rejected because of existing
ticker positions or portfolio-capacity constraints.

### Common-period performance comparison

The common valuation period runs from July 16, 2025, through
November 3, 2026, covering 329 daily portfolio valuations and 328
daily return periods.

| Metric | Random Forest | Momentum | SPY buy-and-hold |
|---|---:|---:|---:|
| Cumulative return | 25.33% | 18.23% | 22.58% |
| CAGR | 18.96% | 13.75% | 16.95% |
| Annualized volatility | 10.69% | 11.10% | 12.30% |
| Sharpe ratio | 1.677 | 1.215 | 1.329 |
| Maximum drawdown | -5.24% | -6.12% | -8.89% |
| Positive trading days | 56.71% | 53.66% | 53.05% |

The Random Forest strategy achieved the highest cumulative return
and Sharpe ratio and the smallest maximum drawdown over this period.
The momentum strategy produced a positive return but underperformed
both Random Forest and SPY.

### Valuation alignment and limitations

Four momentum positions remained open on November 3 and were valued
using that day's closing prices. Their scheduled exits occurred on
November 4. Therefore, the common-period momentum return of 18.23%
differs from the 17.98% return calculated after all 320 trades exited.

Both active strategies use simplified execution assumptions, including
fixed position sizes, fractional shares, adjusted OHLC price proxies,
and no market-impact model. The SPY benchmark does not have fully
matched transaction-cost assumptions.

The historical results are research findings rather than evidence of
reliable future profitability. Additional walk-forward validation,
transaction-cost sensitivity analysis, and testing across market
regimes remain necessary.

### Reproducible artifacts

- `src/pipeline/06_momentum_backtest.py`
- `results/momentum_executed_trades.csv`
- `results/momentum_daily_equity.parquet`
- `results/momentum_performance_metrics.csv`
- `results/marketpulse_three_strategy_comparison.csv`

The alternative momentum script can be executed independently with:

```powershell
python "src/pipeline/06_momentum_backtest.py"
```

The existing five-stage automated pipeline remains unchanged; the
momentum strategy can run independently and is also integrated as Stage 6 of the automated pipeline.


---

# 6. Automation

## 6.1 Automated Pipeline Architecture

MarketPulse AI provides an automated Python pipeline that executes six stages:

| Stage | Script | Purpose |
|---|---|---|
| 1 | `01_data_collection.py` | Collect equity, macroeconomic, and sentiment data |
| 2 | `02_eda_feature_engineering.py` | Clean data and generate engineered features |
| 3 | `03_model_training.py` | Train models and generate predictions |
| 4 | `04_trading_backtest.py` | Execute historical trading simulations |
| 5 | `05_strategy_analysis.py` | Evaluate strategy performance and robustness |
| 6 | `06_momentum_backtest.py` | Execute an independent momentum trading simulation |

The exported scripts are stored in `src/pipeline/`.

## 6.2 Pipeline Orchestrator

The orchestrator is located at `src/run_pipeline.py`.

It executes the pipeline stages sequentially using Python subprocesses.

If a stage fails, execution stops and returns a nonzero exit code.

## 6.3 Pipeline Execution Modes

**Cached mode — uses existing market data**

```powershell
python .\src\run_pipeline.py --mode cached
```

Runs stages 2-6 without downloading new market data.

**Full mode — refreshes market data**

```powershell
python .\src\run_pipeline.py --mode full
```

Runs stages 1-6, including external data collection.

Full mode requires network access and has not yet been validated end to end through the orchestrator.

**Dry-run mode — previews execution**

```powershell
python .\src\run_pipeline.py --mode cached --dry-run
```

Displays the stages that would execute without running them.

## 6.4 Windows Task Scheduler

MarketPulse AI includes a Windows PowerShell launcher:

`scripts/run_marketpulse.ps1`

A local Windows scheduled task has been registered with the following configuration:

| Setting | Value |
|---|---|
| Task name | `MarketPulseAI-Weekly` |
| Frequency | Every Sunday |
| Time | 6:00 PM Central Time |
| Pipeline mode | Cached |
| Maximum execution time | 3 hours |
| Missed-run handling | Start when available |

The task uses the Windows interactive user account and requires an available signed-in session.

The `scripts/setup_schedule.ps1` file prepares and previews the schedule configuration but does not currently register the task.

## 6.5 Logging and Error Handling

The PowerShell launcher executes the pipeline using the project virtual environment.

It sets the Matplotlib backend to `Agg` to prevent interactive plot windows from interrupting automated execution.

Execution output is written to timestamped log files in `logs/`.

The `logs/` directory is excluded from Git.

## 6.6 Automation Validation

The cached pipeline was successfully executed end to end on November 11, 2026, using the existing local market datasets.

The orchestrator completed all five cached-mode stages:

- Stage 02: Feature engineering
- Stage 03: Model training
- Stage 04: Random Forest trading backtest
- Stage 05: Strategy analysis
- Stage 06: Momentum trading backtest

The run finished with the message `Pipeline completed.`

Both cached and full execution modes also passed orchestrator dry-run validation. Full-mode execution has not yet been tested end to end.

The Windows scheduled task was previously registered and reported ready. Its first scheduled execution has not yet been verified.

## 6.7 Automation Limitations

The current automation has the following limitations:

- Weekly cached execution does not download fresh market data.
- Full-mode execution has not yet been tested end to end.
- Incremental data ingestion has not been implemented.
- Windows scheduling depends on computer and user-session availability.
- Automated notifications and cloud scheduling are not implemented.


---

# 7. Reproducibility

## 7.1 Reproducibility Objectives

MarketPulse AI is structured to support repeatable data processing, machine learning, historical backtesting, and strategy analysis.

The project provides Jupyter notebooks, exported Python scripts, a pipeline orchestrator, dependency specifications, and a Streamlit dashboard.

The objective is to allow another developer to reproduce the analytical workflow and evaluate its assumptions and limitations.

## 7.2 System Requirements

The project was developed using:

- Windows and PowerShell
- Python 3.12
- Python virtual environment (`.venv`)
- Jupyter Notebook
- Streamlit

Major Python libraries include pandas, NumPy, scikit-learn, XGBoost, yfinance, statsmodels, Matplotlib, Plotly, and PyArrow.

The project dependencies are recorded in `requirements.txt`.

## 7.3 Clone and Set Up the Environment

Clone the repository:

```powershell
git clone https://github.com/robanc/marketpulse-ai.git
cd marketpulse-ai
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

All subsequent commands should be executed from the project root.

## 7.4 Data Preparation and Storage

MarketPulse AI uses three principal data sources:

1. Yahoo Finance for historical equity market data.
2. FRED for macroeconomic indicators.
3. CNN Fear & Greed Index for market sentiment.

The project stores processed datasets in Parquet format under `data/`.

Examples include:

- `data/stock_prices.parquet`
- `data/fred_macro.parquet`
- `data/market_sentiment.parquet`

The `data/` directory is excluded from Git.

Therefore, a fresh repository clone requires the datasets to be obtained or regenerated before cached execution.

Data collection can be attempted using the full pipeline, subject to external source availability and the current full-mode validation limitations.

## 7.5 Running the Pipeline

To preview execution:

```powershell
python .\src\run_pipeline.py --mode cached --dry-run
```

To execute using existing datasets:

```powershell
python .\src\run_pipeline.py --mode cached
```

To execute all stages, including data collection:

```powershell
python .\src\run_pipeline.py --mode full
```

The cached pipeline requires the previously generated datasets and supporting project artifacts.

Full mode is intended to regenerate source datasets, but has not yet been validated end to end.

## 7.6 Reproducing the Modeling Workflow

The modeling workflow includes:

- Chronological training, validation, and test splits
- Feature engineering and feature availability checks
- Baseline model evaluation
- Machine learning model comparison
- Hyperparameter tuning
- Out-of-sample evaluation
- Model and prediction artifact generation

The modeling methodology and results are documented in Section 4.

Exact numerical reproduction may depend on package versions, data revisions, and external source availability.

## 7.7 Reproducing the Trading Simulation

The trading simulation uses model-generated probabilities, defined trading rules, portfolio allocation assumptions, and transaction cost modeling.

The backtesting methodology is documented in Section 5.

The verified historical simulation produced:

| Metric | Result |
|---|---:|
| Simulated trades | 299 |
| Portfolio observations | 329 |
| Strategy total return | 25.33% |
| SPY benchmark return | 22.58% |
| Strategy maximum drawdown | -5.24% |
| SPY maximum drawdown | -8.88% |

These are historical simulation results, not guarantees of future performance.

## 7.8 Reproducibility Validation

On November 11, 2026, the updated cached pipeline completed successfully in the primary project directory.

Before execution, the existing datasets, trained models, and results were backed up. All 34 backup files passed SHA-256 integrity verification.

After pipeline execution, the regenerated daily equity curves were compared with the original backup artifacts using NumPy `allclose` with relative tolerance `1e-8` and absolute tolerance `1e-6`.

| Verification | Random Forest | Momentum |
|---|---:|---:|
| Valuation dates | 329 | 329 |
| Final simulated equity | $125,326.20 | $118,233.38 |
| Equity curve comparison | Passed | Passed |
| Reproducibility verification | Passed | Passed |

The Random Forest strategy reproduced a 25.33% cumulative simulated return. The momentum strategy reproduced an 18.23% cumulative simulated return.

This confirms reproducibility of both strategy equity curves using the available local datasets and cached execution workflow.

It does not establish that a fresh clone can reproduce every result without additional data preparation or that full-mode execution will succeed under all external data conditions.

## 7.9 Known Limitations

Reproducibility is subject to several constraints:

- Market data may change because of provider corrections or adjustments.
- External APIs and data services may change or become unavailable.
- The repository does not include the ignored local datasets.
- Full-mode automation has not yet been validated end to end.
- Incremental ingestion and persistent pipeline checkpoints are not implemented.
- Historical simulations use assumptions that may differ from live market execution.


---

# 8. Advanced Features and Enhancements

## 8.1 Interactive Streamlit Dashboard

MarketPulse AI includes an interactive analytical dashboard implemented in `app/streamlit_app.py`.

The dashboard organizes project results into five major areas:

| Dashboard Area | Analytical Purpose |
|---|---|
| Portfolio Overview | Review portfolio-level performance |
| Trading Performance | Examine simulated trading results and trade history |
| Portfolio Risk Analysis | Evaluate portfolio risk and drawdown behavior |
| Machine Learning Model Evaluation | Review predictive model performance |
| Strategy Insights | Examine robustness, profit concentration, and key findings |

The application uses Streamlit and Plotly for presenting analytical results.

To launch the dashboard from the project root:

```powershell
streamlit run .\\app\\streamlit_app.py
```

The dashboard is designed to communicate historical research findings rather than execute live trades.

## 8.2 Multi-Source Market Data Integration

MarketPulse AI integrates three complementary information sources:

- Historical equity prices and trading volume from Yahoo Finance.
- Macroeconomic indicators from FRED.
- Market sentiment from the CNN Fear & Greed Index.

Combining these sources supports analysis of technical, macroeconomic, and sentiment-related market conditions.

The underlying data preparation methodology is documented in Sections 2 and 3.

## 8.3 Machine Learning Model Evaluation

The project includes classification and regression model comparison outputs, together with Random Forest hyperparameter tuning results.

Model evaluation considers predictive performance, chronological validation, and the limitations of using historical market features to forecast future outcomes.

The Streamlit application includes a dedicated machine learning evaluation section.

Model selection and evaluation details are documented in Section 4.

## 8.4 Portfolio Risk and Trading Analytics

Beyond total returns, MarketPulse AI evaluates portfolio risk and trading behavior.

The project generates analytical outputs covering:

- Benchmark-relative risk
- Portfolio drawdowns
- Rolling performance
- Monthly performance
- Portfolio exposure
- Position distribution
- Trade-level performance

These analyses provide additional context for interpreting simulated investment performance.

## 8.5 Strategy Robustness and Explainability

The strategy-analysis pipeline generates 13 CSV artifacts under `results/strategy_analysis/`.

These include:

| Analysis | Output File |
|---|---|
| Benchmark risk | `benchmark_risk_summary.csv` |
| Probability calibration | `calibration_by_probability.csv` |
| Calibration metrics | `calibration_metrics.csv` |
| Confidence-band performance | `confidence_band_performance.csv` |
| Drawdown analysis | `drawdown_summary.csv` |
| Monthly performance | `monthly_performance.csv` |
| Portfolio exposure | `portfolio_exposure.csv` |
| Position distribution | `position_distribution.csv` |
| Profit concentration | `profit_concentration.csv` |
| Robustness analysis | `robustness_summary.csv` |
| Rolling performance | `rolling_performance.csv` |
| Ticker-level performance | `ticker_performance.csv` |
| Trade summary | `trade_summary.csv` |

These outputs help assess whether simulated results are concentrated in particular securities, periods, or trading conditions.

They do not establish that the strategy will remain profitable in future markets.

## 8.6 Automated Research Workflow

MarketPulse AI extends the notebook-based research process with five exported Python pipeline scripts and a centralized orchestrator.

The cached workflow has been validated end to end in an isolated project copy.

A Windows scheduled task is configured to execute the cached pipeline weekly.

Automation architecture and scheduling details are documented in Section 6, while reproduction instructions are provided in Section 7.

## 8.7 Future Enhancements

Potential improvements include:

- Validating the complete fresh-data pipeline.
- Implementing incremental market data ingestion.
- Adding automated pipeline health notifications.
- Supporting cloud-based scheduled execution.
- Expanding model monitoring and drift detection.
- Incorporating additional transaction cost and market-impact assumptions.
- Evaluating performance across additional historical market regimes.

These items are proposed enhancements and are not presented as completed functionality.


---

# 9. Results and Key Findings

## 9.1 Executive Results Summary

MarketPulse AI combines historical market data, machine learning, and a simulated trading strategy to evaluate risk-adjusted investment performance.

The strategy achieved a simulated cumulative return of 25.33%, compared with 22.58% for SPY buy-and-hold over the evaluated period.

Although the simulated portfolio showed favorable risk-adjusted performance, predictive model discrimination was weak and profits were highly concentrated.

These findings support further research rather than a conclusion that the strategy has a reliable predictive advantage.

## 9.2 Portfolio Performance Comparison

| Metric | MarketPulse AI | SPY Buy & Hold |
|---|---:|---:|
| Cumulative Return | 25.33% | 22.58% |
| Annualized Return | 18.94% | 16.93% |
| Annualized Volatility | 10.69% | 12.30% |
| Sharpe Ratio | 1.68 | 1.33 |
| Maximum Drawdown | -5.24% | -8.89% |
| Positive Day Rate | 56.71% | 53.05% |
| Trading Days | 328 | 328 |

The strategy exceeded the benchmark by approximately 2.74 percentage points in cumulative return.

Its lower annualized volatility and smaller maximum drawdown contributed to a higher simulated Sharpe ratio.

These comparisons depend on the backtest assumptions and should not be interpreted as live trading performance.

## 9.3 Machine Learning Model Comparison

The classification model comparison produced the following results:

| Model | Accuracy | F1 Score | ROC AUC |
|---|---:|---:|---:|
| Random Forest | 0.5919 | 0.7355 | 0.5690 |
| XGBoost | 0.5692 | 0.7091 | 0.5371 |
| Decision Tree | 0.5777 | 0.7040 | 0.5254 |
| Baseline | 0.5856 | 0.7387 | 0.5000 |

Random Forest achieved the highest ROC AUC in the saved model comparison.

However, the strategy analysis reported a final test ROC AUC of 0.5125 and a model Brier score of 0.2510, worse than its constant-probability baseline.

These results indicate weak out-of-sample predictive discrimination and limited probability calibration quality.

The model comparison and final test evaluation represent different evaluation outputs and should not be treated as interchangeable.

## 9.4 Trading Performance

The historical trading simulation generated the following results:

| Metric | Result |
|---|---:|
| Total Simulated Trades | 299 |
| Winning Trades | 155 |
| Losing Trades | 144 |
| Win Rate | 51.84% |
| Average Net Trade Return | 0.85% |
| Median Net Trade Return | 0.39% |
| Best Trade Return | 29.53% |
| Worst Trade Return | -18.54% |
| Profit Factor | 1.42 |

The profit factor above 1.0 indicates that aggregate simulated winning-trade profits exceeded aggregate losing-trade losses.

The modest win rate and substantial dispersion between the best and worst trades highlight the importance of risk management.

## 9.5 Benchmark Risk Analysis

The benchmark-relative analysis produced:

| Metric | Value |
|---|---:|
| Daily Return Correlation | 0.6364 |
| Portfolio Beta vs SPY | 0.5531 |
| Annualized Tracking Error | 9.91% |
| Annualized Arithmetic Active Return | 1.57% |
| Information Ratio | 0.1590 |
| Downside Capture Ratio | 0.4627 |
| Aligned Daily Observations | 328 |

The portfolio beta below 1.0 indicates lower measured sensitivity to SPY over the evaluated period.

The positive but relatively small information ratio suggests that benchmark-relative excess return was modest relative to tracking error.

## 9.6 Strategy Robustness and Profit Concentration

The strategy robustness analysis identified several important findings:

- Simulated cumulative return was positive.
- Maximum drawdown was smaller than the SPY benchmark.
- The portfolio had lower measured market beta.
- The top 10 trades accounted for approximately 79.30% of net profit.
- The final test ROC AUC was approximately 0.5125.
- The model Brier score was worse than the constant baseline.

High profit concentration means that a small number of trades contributed disproportionately to overall results.

This raises concerns about the stability and repeatability of simulated outperformance.

## 9.7 Key Findings

The project produced five principal findings:

1. The simulated strategy outperformed SPY in cumulative return over the evaluated historical period.
2. Risk-adjusted results were favorable, with lower measured volatility, drawdown, and market beta.
3. Machine learning classification performance remained weak on the final test evaluation.
4. Trading profitability was highly concentrated in a small number of executions.
5. Historical backtest performance alone is insufficient to establish a robust or deployable investment strategy.

## 9.8 Final Assessment

MarketPulse AI demonstrates an end-to-end analytical workflow integrating market data engineering, machine learning, trading simulation, risk evaluation, strategy analysis, and dashboard visualization.

Its strongest contribution is the reproducible research framework and transparent evaluation of both favorable and unfavorable results.

The current evidence does not establish a reliable predictive trading edge.

Further validation, stronger probability calibration, more realistic execution assumptions, and additional out-of-sample testing would be required before considering real-world deployment.


---

# 10. Limitations and Responsible Use

## 10.1 Research Scope and Intended Use

MarketPulse AI is an educational research project developed for the Stock Markets Analytics Zoomcamp 2026 capstone.

It demonstrates the integration of financial data engineering, machine learning, historical trading simulation, portfolio analytics, and dashboard visualization.

The system is intended for research, learning, and analytical evaluation.

It is not a live trading platform, investment advisory service, or production-ready portfolio management system.

## 10.2 Predictive Model Limitations

The final model demonstrated weak out-of-sample predictive performance.

The strategy analysis reported:

| Metric | Result |
|---|---:|
| Final Test ROC AUC | 0.5125 |
| Model Brier Score | 0.2510 |
| Constant Baseline Brier Score | Approximately 0.2481 |

A test ROC AUC close to 0.50 indicates limited ability to distinguish positive and negative outcomes.

The Brier score being worse than the constant baseline indicates that the model's probability predictions did not improve upon that simple benchmark under the evaluated conditions.

Consequently, the available results do not establish a reliable predictive advantage.

## 10.3 Historical Data Limitations

The project relies on historical market data from Yahoo Finance, FRED, and the CNN Fear & Greed Index.

Potential limitations include:

- Historical data corrections and revisions.
- Differences in reporting frequency and publication timing.
- Missing observations and alignment assumptions.
- Corporate actions and adjusted-price conventions.
- Changes to external data providers and APIs.
- Potential survivorship bias from the selected equity universe.

Historical feature availability must be considered carefully to avoid introducing information that would not have been known at the time of a simulated decision.

## 10.4 Backtesting Assumptions

The trading simulation uses historical prices and defined portfolio execution rules.

Its assumptions may differ from real market execution.

Important limitations include:

- Adjusted OHLC prices are used as a historical price proxy.
- Fractional-share and portfolio allocation assumptions may differ from brokerage execution.
- Transaction costs and benchmark cost treatment are not fully equivalent.
- Slippage, market impact, and liquidity constraints are not comprehensively modeled.
- Historical fills do not guarantee executable prices in live markets.
- The selected historical period may not represent future market conditions.

These assumptions can materially affect realized investment performance.

## 10.5 Profit Concentration and Strategy Stability

The strategy generated 299 simulated trades with a win rate of approximately 51.84%.

However, the top 10 trades accounted for approximately 79.30% of net profit.

This concentration suggests that a relatively small number of outcomes contributed disproportionately to overall profitability.

A strategy with concentrated profits may be sensitive to changes in market conditions, execution prices, or individual trade outcomes.

Additional robustness testing is necessary before drawing conclusions about repeatability.

## 10.6 Benchmark Comparison Limitations

MarketPulse AI compares its simulated portfolio against SPY buy-and-hold.

The strategy achieved a higher historical cumulative return and smaller maximum drawdown during the evaluated period.

However, the comparison is affected by differences in exposure, trading activity, allocation, transaction costs, and execution assumptions.

A favorable historical benchmark comparison does not establish superior future performance.

## 10.7 Automation and Operational Limitations

The project includes a Python orchestrator and a registered Windows scheduled task.

The cached pipeline has been validated in an isolated environment, but the scheduled task has not yet completed its first scheduled execution.

Operational limitations include:

- Cached runs do not automatically refresh external market data.
- Full-mode execution has not yet been validated end to end.
- Incremental ingestion and checkpoint recovery are not implemented.
- Scheduled execution depends on local computer and Windows session availability.
- Automated alerting and cloud-based monitoring are not implemented.

The automation should therefore be treated as a local research workflow rather than a production data platform.

## 10.8 Responsible Interpretation of Results

MarketPulse AI results should be interpreted as historical simulation outputs.

Users should not assume that:

- Positive historical returns will continue.
- A higher historical Sharpe ratio guarantees future risk-adjusted outperformance.
- Machine learning predictions are reliable investment signals.
- Simulated trade execution reflects actual brokerage execution.
- The strategy is suitable for a particular investor's financial circumstances.

Financial decisions require consideration of individual objectives, risk tolerance, liquidity needs, and independent due diligence.

## 10.9 Future Validation Requirements

Before considering real-world deployment, additional work would be necessary.

Recommended validation includes:

1. Testing across additional out-of-sample periods and market regimes.
2. Improving probability calibration and comparing against simple baselines.
3. Evaluating sensitivity to transaction costs, slippage, and liquidity.
4. Performing walk-forward testing and parameter stability analysis.
5. Assessing survivorship bias and point-in-time data availability.
6. Validating the full data refresh pipeline.
7. Implementing monitoring, error notifications, and operational safeguards.
8. Conducting paper trading before any consideration of live execution.

## 10.10 Disclaimer

MarketPulse AI is provided for educational and research purposes only.

The project does not provide personalized financial advice, investment recommendations, or guarantees of returns.

All reported trading results are historical simulations.

Past performance does not guarantee future results.


---

# 11. References and Acknowledgments

## 11.1 Educational Program

MarketPulse AI was developed as a capstone project for the Stock Markets Analytics Zoomcamp 2026.

The project applies concepts from the program, including financial data collection, exploratory analysis, machine learning, portfolio evaluation, and investment strategy simulation.

Educational resource:

- [DataTalks.Club](https://datatalks.club/)

## 11.2 Financial Data Sources

The project uses the following financial data providers and resources:

| Source | Purpose | Reference |
|---|---|---|
| Yahoo Finance | Historical equity market data | https://finance.yahoo.com/ |
| FRED | Macroeconomic indicators | https://fred.stlouisfed.org/ |
| CNN Fear & Greed Index | Market sentiment data | https://www.cnn.com/markets/fear-and-greed |

Historical market data may be revised, and provider access or data formats may change over time.

## 11.3 Python Libraries and Frameworks

Major technologies used in the project include:

| Technology | Purpose | Reference |
|---|---|---|
| Python | Core programming language | https://www.python.org/ |
| pandas | Data processing and analysis | https://pandas.pydata.org/ |
| NumPy | Numerical computation | https://numpy.org/ |
| scikit-learn | Machine learning and evaluation | https://scikit-learn.org/ |
| XGBoost | Gradient-boosted machine learning | https://xgboost.readthedocs.io/ |
| yfinance | Yahoo Finance data access | https://github.com/ranaroussi/yfinance |
| statsmodels | Statistical modeling | https://www.statsmodels.org/ |
| Matplotlib | Data visualization | https://matplotlib.org/ |
| Plotly | Interactive visualization | https://plotly.com/python/ |
| Streamlit | Interactive analytical dashboard | https://streamlit.io/ |
| PyArrow | Parquet data storage | https://arrow.apache.org/docs/python/ |

The complete Python dependency specification is maintained in `requirements.txt`.

## 11.4 Project Repository

Source code, notebooks, exported pipeline scripts, and project documentation are maintained in the GitHub repository:

https://github.com/robanc/marketpulse-ai

The repository includes the implementation of the data processing, modeling, backtesting, analysis, dashboard, and local automation workflows.

## 11.5 Acknowledgments

Acknowledgment is given to DataTalks.Club and the Stock Markets Analytics Zoomcamp learning community for providing the educational framework and project guidance.

The project also benefits from the open-source Python ecosystem and publicly accessible financial data resources used throughout the analytical workflow.

## 11.6 Research and Financial Disclaimer

References to external data providers, software libraries, and educational resources are provided for attribution and reproducibility.

Their inclusion does not imply endorsement of MarketPulse AI or its simulated trading results.

MarketPulse AI is an educational research project and does not provide financial advice or guarantees of investment performance.
