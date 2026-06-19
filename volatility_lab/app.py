import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from volatility_lab import TRADING_DAYS
from volatility_lab.backtesting import (
    annualized_vol_to_daily_var,
    confidence_band,
    diebold_mariano_table,
    eval_models,
    forecast_error_frame,
    kupiec_var_test,
    rolling_model_ranking,
)
from volatility_lab.data import (
    fetch_yfinance_daily,
    fetch_yfinance_intraday,
    read_csv_bytes,
    sample_daily_data,
    to_log_returns_from_prices,
    validate_price_frame,
)
from volatility_lab.models import ewma_volatility, har_rv_forecast, rolling_garch_forecast, stacked_hybrid_forecast
from volatility_lab.plotting import correlation_heatmap, plot_series, realized_forecast_scatter
from volatility_lab.portfolio import (
    inverse_vol_weights,
    make_multi_asset_returns,
    minimum_variance_weights,
    parametric_var_1d,
    portfolio_pnl,
    portfolio_summary,
    risk_contributions,
    vol_target_positions,
)
from volatility_lab.realized_vol import (
    aggregate_intraday_realized_vol,
    realized_var_from_daily_returns,
    realized_vol_from_daily_returns,
)
from volatility_lab.reporting import download_dataframe, html_report, make_run_output_dir, save_run_tables

warnings.filterwarnings("ignore")


@st.cache_data(show_spinner=False)
def cached_yfinance_daily(ticker: str, start_date, end_date) -> pd.DataFrame:
    return fetch_yfinance_daily(ticker, start_date, end_date)


@st.cache_data(show_spinner=False)
def cached_yfinance_intraday(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return fetch_yfinance_intraday(ticker, period, interval)


@st.cache_data(show_spinner=False)
def cached_read_csv(data: bytes) -> pd.DataFrame:
    return read_csv_bytes(data)


@st.cache_data(show_spinner=False)
def cached_daily_realized(returns: pd.Series, window: int):
    return realized_vol_from_daily_returns(returns, window), realized_var_from_daily_returns(returns, window)


@st.cache_data(show_spinner=False)
def cached_intraday_realized(
    data: bytes, columns: tuple[str, str, str, str, str], estimator: str, kernel_lag: int
) -> pd.DataFrame:
    intraday = read_csv_bytes(data)
    return aggregate_intraday_realized_vol(intraday, *columns, estimator, kernel_lag)


@st.cache_data(show_spinner=False)
def cached_ewma(returns: pd.Series, lam: float) -> pd.Series:
    return ewma_volatility(returns, lam)


@st.cache_data(show_spinner=False)
def cached_har(rv_vol: pd.Series, h: int, ridge_alpha: float, train_window: int, expanding: bool) -> pd.Series:
    preds, _ = har_rv_forecast(rv_vol, h, ridge_alpha, train_window, expanding)
    return preds


@st.cache_data(show_spinner=False)
def cached_hybrid(rv_vol: pd.Series, base_forecasts: dict, train_window: int, ridge_alpha: float) -> pd.Series:
    return stacked_hybrid_forecast(rv_vol, base_forecasts, train_window, ridge_alpha)


def apply_mode_defaults(mode: str) -> dict:
    if mode == "Fast mode":
        return {"use_garch": False, "use_egarch": False, "use_gjr": False, "train_window": 504, "max_forecasts": 250}
    if mode == "Research mode":
        return {"use_garch": True, "use_egarch": True, "use_gjr": True, "train_window": 1000, "max_forecasts": 1000}
    return {"use_garch": True, "use_egarch": False, "use_gjr": False, "train_window": 756, "max_forecasts": 500}


def model_settings_dict(**kwargs) -> dict:
    return {key: value for key, value in kwargs.items()}


def run_garch_with_progress(
    returns: pd.Series,
    selected: list[str],
    train_window: int,
    horizon: int,
    dist: str,
    expanding: bool,
    max_forecasts: int,
) -> dict:
    outputs = {}
    total_steps = len(selected)
    progress = st.progress(0, text="Preparing GARCH fits...")
    for model_pos, variant in enumerate(selected):

        def update(done, total, variant=variant, model_pos=model_pos):
            completed = (model_pos + done / max(total, 1)) / max(total_steps, 1)
            progress.progress(min(completed, 1.0), text=f"Running {variant}: {done}/{total} rolling fits")

        outputs[f"{variant}11_h{horizon}"] = rolling_garch_forecast(
            returns,
            train_window=train_window,
            horizon=horizon,
            variant=variant,
            dist=dist,
            expanding=expanding,
            max_forecasts=max_forecasts,
            progress_callback=update,
        )
    progress.empty()
    return outputs


def main():
    st.set_page_config(page_title="Volatility Lab", layout="wide")
    st.title("Volatility Forecasting Lab")
    st.markdown(
        """
Upload daily prices, optional intraday OHLC bars, or a wide multi-asset price file.
The lab forecasts volatility, evaluates models with variance-aware losses, and routes forecasts into risk controls.
"""
    )

    with st.sidebar:
        st.header("Data")
        data_provider = st.selectbox(
            "Daily data source", ["Yahoo Finance", "CSV upload", "Kaggle/local CSV", "Sample dataset"], index=0
        )
        uploaded = st.file_uploader("Upload daily price CSV", type=["csv"])
        ticker = st.text_input("Ticker / symbol", value="SPY")
        source_start = st.date_input("Source start date", value=pd.Timestamp("2010-01-01").date())
        source_end = st.date_input("Source end date", value=pd.Timestamp.today().date())
        fetch_yf_intraday = st.checkbox("Fetch Yahoo intraday bars", value=False)
        yf_intraday_period = st.selectbox("Yahoo intraday period", ["5d", "1mo", "3mo", "6mo", "60d"], index=4)
        yf_intraday_interval = st.selectbox(
            "Yahoo intraday interval", ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"], index=2
        )
        intraday_uploaded = st.file_uploader("Optional intraday OHLC CSV", type=["csv"])
        multi_uploaded = st.file_uploader("Optional multi-asset daily prices CSV", type=["csv"])

        st.header("Columns")
        date_col = st.text_input("Date column", value="Date")
        price_col = st.text_input("Price column", value="Close")
        intraday_dt_col = st.text_input("Intraday datetime column", value="Datetime")
        intraday_open_col = st.text_input("Intraday open column", value="Open")
        intraday_high_col = st.text_input("Intraday high column", value="High")
        intraday_low_col = st.text_input("Intraday low column", value="Low")
        intraday_close_col = st.text_input("Intraday close column", value="Close")

        st.header("Runtime Mode")
        mode = st.selectbox("Run profile", ["Fast mode", "Full mode", "Research mode"], index=0)
        defaults = apply_mode_defaults(mode)
        skip_garch_rows = st.number_input(
            "Skip GARCH if rows exceed", min_value=500, max_value=20000, value=6000, step=500
        )
        max_forecasts = st.slider("Max rolling GARCH forecasts", 50, 1500, defaults["max_forecasts"], step=50)

        st.header("Realized Volatility")
        rv_window = st.slider("Daily rolling window", 5, 60, 21)
        intraday_estimator = st.selectbox(
            "Intraday estimator", ["Realized variance", "Parkinson", "Garman-Klass", "Yang-Zhang", "Realized kernel"]
        )
        kernel_lag = st.slider("Realized kernel lag", 1, 10, 3)

        st.header("Models")
        use_ewma = st.checkbox("EWMA", value=True)
        lam = st.slider("EWMA lambda", 0.80, 0.99, 0.94)
        use_garch = st.checkbox("GARCH(1,1)", value=defaults["use_garch"])
        use_egarch = st.checkbox("EGARCH(1,1)", value=defaults["use_egarch"])
        use_gjr = st.checkbox("GJR-GARCH(1,1)", value=defaults["use_gjr"])
        train_window = st.slider("GARCH train window", 200, 1500, defaults["train_window"])
        expanding_garch = st.checkbox("Expanding-window GARCH", value=False)
        horizon = st.slider("Forecast horizon", 1, 10, 1)
        dist = st.selectbox("GARCH distribution", ["t", "normal"])

        st.header("HAR and Hybrid")
        use_har = st.checkbox("HAR-RV", value=True)
        har_h = st.slider("HAR horizon", 1, 10, 1)
        har_train_window = st.slider("HAR train window", 60, 1000, 252)
        expanding_har = st.checkbox("Expanding-window HAR", value=False)
        har_ridge = st.slider("HAR ridge alpha", 0.0, 50.0, 0.0)
        use_hybrid = st.checkbox("Hybrid stacking", value=True)
        hybrid_window = st.slider("Hybrid train window", 100, 1000, 252)
        hybrid_ridge = st.slider("Hybrid ridge alpha", 0.1, 50.0, 1.0)

        st.header("Backtesting")
        split_mode = st.selectbox(
            "Evaluation split",
            ["Use all aligned dates", "Holdout by fraction", "Holdout by date", "Manual start date"],
            index=1,
        )
        holdout_fraction = st.slider("Holdout fraction", 0.10, 0.50, 0.25)
        holdout_date = st.date_input("Holdout start date", value=pd.Timestamp.today().date())
        eval_start = st.date_input("Manual evaluation start", value=pd.Timestamp("2015-01-01").date())
        rank_window = st.slider("Rolling rank window", 20, 252, 63)

        st.header("Portfolio / Risk")
        target_vol = st.slider("Target annual volatility", 0.05, 0.50, 0.15)
        max_lev = st.slider("Max leverage", 0.5, 5.0, 2.0)
        var_z = st.slider("VaR z-score", 1.0, 3.5, 2.33)
        max_asset_weight = st.slider("Max asset weight", 0.10, 1.00, 0.35)
        st.header("Output Saving")
        save_outputs = st.checkbox("Save run outputs", value=False)
        output_base_dir = st.text_input("Output folder", value="outputs")
        run_btn = st.button("Run Vol Lab")

    if data_provider in ["CSV upload", "Kaggle/local CSV"] and uploaded is None:
        st.info("Upload a daily prices CSV, switch to Yahoo Finance, or use the sample dataset.")
        st.stop()

    try:
        if data_provider == "Yahoo Finance":
            df_raw = cached_yfinance_daily(ticker, source_start, source_end)
            date_col = "Date"
            price_col = "Close"
        elif data_provider in ["CSV upload", "Kaggle/local CSV"]:
            df_raw = cached_read_csv(uploaded.getvalue())
        else:
            df_raw = sample_daily_data()
        df = validate_price_frame(df_raw, date_col, price_col)
        returns = to_log_returns_from_prices(df, price_col)
    except Exception as exc:
        st.error(f"Could not load daily data from {data_provider}: {exc}")
        st.stop()

    rv_source = f"{rv_window}-day rolling daily returns"
    intraday_rv = None
    try:
        if intraday_uploaded is not None:
            intraday_rv = cached_intraday_realized(
                intraday_uploaded.getvalue(),
                (intraday_dt_col, intraday_open_col, intraday_high_col, intraday_low_col, intraday_close_col),
                intraday_estimator,
                kernel_lag,
            )
            rv_var = intraday_rv["realized_var"]
            rv_vol = intraday_rv["realized_vol"]
            rv_source = f"intraday {intraday_estimator}"
        elif fetch_yf_intraday:
            intraday_df = cached_yfinance_intraday(ticker, yf_intraday_period, yf_intraday_interval)
            intraday_rv = aggregate_intraday_realized_vol(
                intraday_df, "Datetime", "Open", "High", "Low", "Close", intraday_estimator, kernel_lag
            )
            rv_var = intraday_rv["realized_var"]
            rv_vol = intraday_rv["realized_vol"]
            rv_source = f"Yahoo {yf_intraday_interval} intraday {intraday_estimator}"
        else:
            rv_vol, rv_var = cached_daily_realized(returns, rv_window)
    except Exception as exc:
        st.warning(f"Intraday target could not be used: {exc}")
        rv_vol, rv_var = cached_daily_realized(returns, rv_window)

    rv_var = rv_var.reindex(rv_vol.index).dropna()
    rv_vol = rv_vol.reindex(rv_var.index).dropna()
    if rv_vol.empty:
        st.error("No realized-volatility target could be computed. Use more rows or a smaller rolling window.")
        st.stop()

    st.subheader("Data Preview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Price rows", f"{len(df):,}")
    c2.metric("Return rows", f"{len(returns):,}")
    c3.metric("Realized-vol target", rv_source)
    st.dataframe(df.tail(5), use_container_width=True)
    st.write("Returns summary")
    st.dataframe(returns.describe().to_frame("logret"), use_container_width=True)

    if intraday_rv is not None:
        st.subheader("Intraday Realized Volatility Targets")
        if len(intraday_rv) < har_train_window + har_h + 25 and use_har:
            st.warning(
                "HAR-RV needs more realized-volatility target rows than the current intraday sample provides. "
                "Use a smaller HAR train window, disable HAR-RV, or use the daily rolling target for a longer history."
            )
        st.dataframe(
            intraday_rv[
                ["Realized variance", "Parkinson", "Garman-Klass", "Yang-Zhang", "Realized kernel", "realized_vol"]
            ].tail(10),
            use_container_width=True,
        )

    garch_variants = []
    if use_garch:
        garch_variants.append("GARCH")
    if use_egarch:
        garch_variants.append("EGARCH")
    if use_gjr:
        garch_variants.append("GJR")
    if len(returns) > skip_garch_rows and garch_variants:
        st.warning(
            "GARCH models were disabled because the row count exceeds the runtime limit. Increase the limit or use a later start date."
        )
        garch_variants = []
    if len(returns) < train_window + horizon + 10 and garch_variants:
        st.warning(
            "The selected GARCH train window is too large for the available data. Reduce it or disable GARCH variants."
        )

    if not run_btn:
        st.stop()

    model_vols = {}
    try:
        with st.spinner("Running volatility models..."):
            if use_ewma:
                model_vols[f"EWMA({lam})"] = cached_ewma(returns, lam)
            if garch_variants:
                model_vols.update(
                    run_garch_with_progress(
                        returns, garch_variants, train_window, horizon, dist, expanding_garch, max_forecasts
                    )
                )
            har_preds = None
            if use_har:
                try:
                    har_preds = cached_har(rv_vol, har_h, har_ridge, har_train_window, expanding_har)
                    model_vols[f"HAR_RV_h{har_h}"] = har_preds
                except ValueError as exc:
                    st.warning(f"HAR-RV skipped: {exc}. Try a smaller HAR train window or more data.")
            if use_hybrid:
                bases = {name: vol for name, vol in model_vols.items() if name != "HYBRID_STACKED"}
                if len(bases) >= 2:
                    try:
                        model_vols["HYBRID_STACKED"] = cached_hybrid(rv_vol, bases, hybrid_window, hybrid_ridge)
                    except ValueError as exc:
                        st.warning(f"Hybrid stacking skipped: {exc}. Try a smaller hybrid train window or more data.")
                else:
                    st.warning("Hybrid stacking needs at least two base forecasts.")
    except Exception as exc:
        st.error(f"Model run failed: {exc}")
        st.stop()

    if not model_vols:
        st.error("No models were selected or able to run.")
        st.stop()

    aligned_index = rv_vol.index
    for vol in model_vols.values():
        aligned_index = aligned_index.intersection(vol.dropna().index)

    test_start = None
    if split_mode == "Holdout by fraction" and len(aligned_index) > 5:
        test_start = aligned_index[int(len(aligned_index) * (1 - holdout_fraction))]
    elif split_mode == "Holdout by date":
        test_start = pd.Timestamp(holdout_date)
    elif split_mode == "Manual start date":
        test_start = pd.Timestamp(eval_start)

    metrics_df = eval_models(rv_vol, rv_var, model_vols, test_start)
    if metrics_df.empty:
        st.error("No forecasts aligned with the realized-volatility target.")
        st.stop()

    best_model_name = metrics_df.iloc[0]["model"]
    best_forecast_vol = model_vols[best_model_name].dropna()
    run_dir = make_run_output_dir(output_base_dir, ticker) if save_outputs else None

    def plot_path(filename: str):
        return run_dir / "plots" / filename if run_dir is not None else None

    st.subheader("Model Comparison Dashboard")
    st.dataframe(metrics_df, use_container_width=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best model", best_model_name)
    m2.metric("Best QLIKE", f"{metrics_df.iloc[0]['QLIKE(var)']:.4f}")
    m3.metric("Best RMSE", f"{metrics_df.iloc[0]['RMSE(vol)']:.4f}")
    m4.metric("Evaluation start", str(test_start.date()) if test_start is not None else "All aligned")
    st.info(
        f"{best_model_name} ranks first by QLIKE. Lower QLIKE means better variance forecast quality on the selected evaluation window."
    )

    st.subheader("Forecasts vs Realized Volatility")
    plot_series(
        "Annualized Volatility Forecasts",
        {"Realized": rv_vol, **{k: v.reindex(rv_vol.index) for k, v in model_vols.items()}},
        4,
        save_path=plot_path("annualized_volatility_forecasts.png"),
    )

    lower, upper = confidence_band(rv_vol, best_forecast_vol)
    band = pd.DataFrame({"Realized": rv_vol, "Forecast": best_forecast_vol, "Lower": lower, "Upper": upper}).dropna()
    if not band.empty:
        plot_series(
            "Best Forecast with Confidence Band",
            {
                "Realized": band["Realized"],
                "Forecast": band["Forecast"],
                "Lower": band["Lower"],
                "Upper": band["Upper"],
            },
            4,
            save_path=plot_path("best_forecast_confidence_band.png"),
        )

    st.subheader("Variance View")
    top_models = metrics_df["model"].head(min(3, len(metrics_df))).tolist()
    var_plot = {"Realized variance": rv_var}
    for model in top_models:
        var_plot[model] = annualized_vol_to_daily_var(model_vols[model]).reindex(rv_var.index)
    plot_series("Daily Variance Forecasts", var_plot, 4, save_path=plot_path("daily_variance_forecasts.png"))

    st.subheader("Backtesting Diagnostics")
    errors = forecast_error_frame(rv_vol, model_vols)
    if not errors.empty:
        plot_series(
            "Forecast Errors",
            {col: errors[col] for col in errors.columns},
            4,
            save_path=plot_path("forecast_errors.png"),
        )
        realized_forecast_scatter(
            rv_vol,
            best_forecast_vol,
            save_path=plot_path("realized_vs_forecast_scatter.png"),
        )
    ranks = rolling_model_ranking(rv_var, model_vols, rank_window)
    if not ranks.empty:
        plot_series(
            "Rolling Model Rank by QLIKE (1 = best)",
            {col: ranks[col] for col in ranks.columns},
            4,
            save_path=plot_path("rolling_model_rank_qlike.png"),
        )
    dm_df = diebold_mariano_table(rv_var, model_vols)
    if not dm_df.empty:
        st.write(
            "Diebold-Mariano tests use QLIKE loss differences. Negative mean difference means model A had lower average loss than model B."
        )
        st.dataframe(dm_df.sort_values("p_value"), use_container_width=True)

    st.subheader("Single-Asset Portfolio / Risk")
    weights = vol_target_positions(returns, best_forecast_vol, target_vol, max_lev, lag=1)
    pnl = portfolio_pnl(returns, weights)
    equity = (1 + pnl).cumprod()
    var_1d = parametric_var_1d(best_forecast_vol, var_z).shift(1)
    ann_ret = (equity.iloc[-1] ** (TRADING_DAYS / len(equity)) - 1) if len(equity) > 10 else np.nan
    ann_vol = pnl.std() * np.sqrt(TRADING_DAYS) if len(pnl) > 10 else np.nan
    sharpe = pnl.mean() / pnl.std() * np.sqrt(TRADING_DAYS) if pnl.std() > 0 else np.nan
    max_dd = ((equity / equity.cummax()) - 1).min() if len(equity) else np.nan
    var_stats = kupiec_var_test(pnl, var_1d.reindex(pnl.index), alpha=max(1e-6, 1 - 0.99))

    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Ann Return", f"{ann_ret:.2%}" if np.isfinite(ann_ret) else "NA")
    p2.metric("Ann Vol", f"{ann_vol:.2%}" if np.isfinite(ann_vol) else "NA")
    p3.metric("Sharpe", f"{sharpe:.2f}" if np.isfinite(sharpe) else "NA")
    p4.metric("Max Drawdown", f"{max_dd:.2%}" if np.isfinite(max_dd) else "NA")
    p5.metric("VaR Hit Ratio", f"{var_stats['hit_ratio']:.2%}" if np.isfinite(var_stats["hit_ratio"]) else "NA")
    plot_series(
        "Vol-Targeted Equity Curve", {"Equity": equity}, 4, save_path=plot_path("vol_targeted_equity_curve.png")
    )
    plot_series("Position Weights", {"Weight": weights}, 3, save_path=plot_path("position_weights.png"))
    plot_series(
        "1-Day VaR vs Absolute Return",
        {"VaR_1d": var_1d.reindex(pnl.index), "Abs Return": pnl.abs()},
        3,
        save_path=plot_path("var_vs_absolute_return.png"),
    )
    st.dataframe(pd.DataFrame([var_stats]), use_container_width=True)

    if multi_uploaded is not None:
        st.subheader("Multi-Asset Volatility and Portfolio Risk")
        try:
            multi_returns = make_multi_asset_returns(cached_read_csv(multi_uploaded.getvalue()), date_col)
            asset_vols = multi_returns.rolling(rv_window).std().iloc[-1] * np.sqrt(TRADING_DAYS)
            cov = multi_returns.cov() * TRADING_DAYS
            corr = multi_returns.corr()
            min_var_w = minimum_variance_weights(cov, max_asset_weight)
            risk_parity_w = inverse_vol_weights(asset_vols, max_asset_weight)
            weights_df = pd.DataFrame({"min_variance": min_var_w, "risk_parity_inverse_vol": risk_parity_w}).fillna(0)
            st.write("Per-asset annualized volatility forecasts")
            st.dataframe(asset_vols.rename("forecast_vol").to_frame(), use_container_width=True)
            correlation_heatmap(corr, save_path=plot_path("multi_asset_correlation_heatmap.png"))
            st.write("Portfolio weights")
            st.dataframe(weights_df, use_container_width=True)
            choice = st.selectbox("Portfolio to inspect", weights_df.columns.tolist())
            multi_pnl, stats = portfolio_summary(multi_returns, weights_df[choice], target_vol, var_z)
            rc = risk_contributions(weights_df[choice], cov)
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Ann Return", f"{stats['ann_return']:.2%}" if np.isfinite(stats["ann_return"]) else "NA")
            q2.metric("Ann Vol", f"{stats['ann_vol']:.2%}" if np.isfinite(stats["ann_vol"]) else "NA")
            q3.metric("VaR 1d", f"{stats['var_1d']:.2%}" if np.isfinite(stats["var_1d"]) else "NA")
            q4.metric("CVaR 1d", f"{stats['cvar_1d']:.2%}" if np.isfinite(stats["cvar_1d"]) else "NA")
            st.write("Marginal risk contribution")
            st.dataframe(rc.to_frame(), use_container_width=True)
            plot_series(
                "Multi-Asset Vol-Targeted Equity Curve",
                {"Equity": (1 + multi_pnl).cumprod()},
                4,
                save_path=plot_path("multi_asset_equity_curve.png"),
            )
        except Exception as exc:
            st.warning(f"Multi-asset analysis could not run: {exc}")

    st.subheader("Exports")
    forecast_export = pd.DataFrame({name: vol for name, vol in model_vols.items()})
    forecast_export["realized_vol"] = rv_vol
    forecast_export["realized_var"] = rv_var
    settings = model_settings_dict(
        mode=mode,
        ticker=ticker,
        data_provider=data_provider,
        rv_source=rv_source,
        ewma_lambda=lam,
        garch_variants=", ".join(garch_variants),
        train_window=train_window,
        horizon=horizon,
        max_forecasts=max_forecasts,
        har_horizon=har_h,
        har_train_window=har_train_window,
        hybrid_window=hybrid_window,
        target_vol=target_vol,
        max_leverage=max_lev,
    )
    dataset_summary = {
        "price_rows": len(df),
        "return_rows": len(returns),
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "source": data_provider,
    }
    e1, e2, e3 = st.columns(3)
    with e1:
        download_dataframe(st, "Download metrics CSV", metrics_df, "vol_lab_metrics.csv")
    with e2:
        download_dataframe(st, "Download forecasts CSV", forecast_export, "vol_lab_forecasts.csv")
    with e3:
        st.download_button(
            "Download HTML report",
            html_report(metrics_df, best_model_name, rv_source, settings, dataset_summary),
            "vol_lab_report.html",
            "text/html",
        )

    if run_dir is not None:
        save_run_tables(run_dir, metrics_df, forecast_export, settings, dataset_summary, best_model_name, rv_source)
        risk_frame = pd.DataFrame(
            {
                "returns": returns,
                "weight": weights,
                "pnl": pnl,
                "equity": equity,
                "var_1d": var_1d,
                "abs_return": pnl.abs(),
            }
        )
        risk_frame.to_csv(run_dir / "data" / "single_asset_risk_series.csv", index=True)
        pd.DataFrame([var_stats]).to_csv(run_dir / "data" / "var_backtest.csv", index=False)
        if not errors.empty:
            errors.to_csv(run_dir / "data" / "forecast_errors.csv", index=True)
        if not ranks.empty:
            ranks.to_csv(run_dir / "data" / "rolling_model_ranks.csv", index=True)
        if not dm_df.empty:
            dm_df.to_csv(run_dir / "data" / "diebold_mariano.csv", index=False)
        st.success(f"Saved run outputs to: {run_dir.resolve()}")

    st.caption(
        "Intraday realized volatility is most useful when bars are clean and consistently spaced. "
        "The realized-kernel estimator here is a lightweight Bartlett-kernel approximation for noisy high-frequency returns."
    )


if __name__ == "__main__":
    main()
