# Volatility Forecasting Lab

An end-to-end Streamlit research app for volatility forecasting, realized-volatility benchmarking, forecast backtesting, and portfolio risk controls.

## Project Structure

```text
volatility_lab/
  app.py            # Streamlit UI and cached orchestration
  data.py           # CSV/Yahoo loading and validation
  realized_vol.py   # Daily and intraday realized-volatility targets
  models.py         # EWMA, GARCH-family, HAR-RV, hybrid stacking
  backtesting.py    # QLIKE, DM tests, rankings, VaR diagnostics
  portfolio.py      # Vol targeting and portfolio risk helpers
  plotting.py       # Streamlit/matplotlib chart helpers
  reporting.py      # CSV and HTML report exports
tests/
  test_realized_vol.py
  test_alignment.py
examples/
  daily_prices_sample.csv
  intraday_ohlc_sample.csv
  multi_asset_sample.csv
```

## What It Does

- Upload daily price data with `Date` and `Close` columns.
- Optionally upload intraday OHLC bars, such as 5-minute data, to build better realized-volatility targets.
- Compare volatility models:
  - EWMA
  - GARCH(1,1)
  - EGARCH(1,1)
  - GJR-GARCH(1,1)
  - HAR-RV
  - hybrid stacked ensemble
- Evaluate forecasts with:
  - MAE on annualized volatility
  - RMSE on annualized volatility
  - MSE on daily variance
  - QLIKE on daily variance
  - Diebold-Mariano tests
  - rolling model rankings
  - forecast error plots
  - empirical confidence bands
- Run single-asset risk controls:
  - volatility targeting
  - 1-day parametric VaR
  - equity curve, leverage, drawdown, and Sharpe diagnostics
- Run optional multi-asset portfolio analysis:
  - per-asset volatility estimates
  - forecast covariance matrix
  - correlation heatmap
  - minimum variance portfolio
  - inverse-vol risk parity proxy
  - portfolio VaR and CVaR
  - marginal risk contribution
- Export:
  - metrics CSV
  - forecast CSV
  - HTML report with dataset summary and model settings

## Runtime Modes

The sidebar includes three run profiles:

- **Fast mode**: EWMA + HAR-RV by default. Best for quick exploration and demos.
- **Full mode**: enables GARCH(1,1) with capped rolling forecasts.
- **Research mode**: enables GARCH, EGARCH, and GJR-GARCH with larger windows and higher runtime limits.

Rolling GARCH models are expensive because each forecast refits an `arch` model. Use the `Max rolling GARCH forecasts` and `Skip GARCH if rows exceed` controls to avoid long Streamlit runs.

## Input Formats

The app can load daily prices from:

- Yahoo Finance through `yfinance`
- a local CSV upload
- Kaggle datasets downloaded as CSV files and uploaded through the app
- the built-in sample dataset

Yahoo Finance is the default source. Enter a ticker such as `SPY`, `AAPL`, `MSFT`, or `NVDA`, choose a start/end date, and run the lab. Kaggle datasets should be downloaded locally as CSV files and uploaded through the CSV flow.

For a quick intraday demo, enable `Fetch Yahoo intraday bars` in the sidebar and choose an interval such as `5m`. Yahoo intraday data has limited lookback, so use it as a convenience demo rather than a research-grade long-history source.

Daily price CSV:

```text
Date,Close
2024-01-02,100.25
2024-01-03,101.10
```

Intraday OHLC CSV:

```text
Datetime,Open,High,Low,Close
2024-01-02 09:30:00,100.00,100.20,99.95,100.10
2024-01-02 09:35:00,100.10,100.35,100.05,100.22
```

Multi-asset daily price CSV:

```text
Date,AAPL,MSFT,NVDA,SPY
2024-01-02,185.64,370.87,48.17,472.65
2024-01-03,184.25,370.60,47.57,468.79
```

## Intraday Realized Volatility Estimators

The app can aggregate intraday bars into daily realized-volatility targets using:

- realized variance from intraday returns
- Parkinson volatility
- Garman-Klass volatility
- Yang-Zhang volatility
- a lightweight Bartlett-kernel realized-kernel approximation

These targets make GARCH, EWMA, HAR-RV, and hybrid model comparisons more credible than using only rolling daily variance.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run vol_lab.py
```

Or run the package entrypoint directly:

```bash
streamlit run volatility_lab/app.py
```

You can start immediately by enabling the built-in sample daily dataset in the sidebar.

## Testing and Formatting

Run the checks locally:

```bash
python -m py_compile vol_lab.py volatility_lab/*.py
pytest
ruff check .
black --check .
```

GitHub Actions runs compile and test checks on push and pull request.

## Screenshots

Recommended README images after launching the app:

- `docs/screenshot-dashboard.png`: model comparison dashboard
- `docs/screenshot-risk.png`: single-asset risk controls
- `docs/screenshot-report.png`: export section

Capture these from a representative run using the sample dataset or a ticker such as `SPY`.

## Notes

This is a research and educational tool, not financial advice or a live-trading system. Results depend heavily on data quality, lookahead-safe alignment, model-window choices, and market regime.

The HAR-RV and hybrid forecasts are indexed by the realized-volatility date they predict, so model comparisons align forecasts against the target date rather than the feature-generation date.
