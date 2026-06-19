import math

import numpy as np
import pandas as pd

from . import TRADING_DAYS


def realized_vol_from_daily_returns(returns: pd.Series, window: int = 21) -> pd.Series:
    out = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    out.name = f"daily_rolling_vol_{window}"
    return out.dropna()


def realized_var_from_daily_returns(returns: pd.Series, window: int = 21) -> pd.Series:
    out = returns.rolling(window).var()
    out.name = f"daily_rolling_var_{window}"
    return out.dropna()


def aggregate_intraday_realized_vol(
    intraday: pd.DataFrame,
    datetime_col: str,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    estimator: str,
    kernel_lag: int = 3,
) -> pd.DataFrame:
    required = [datetime_col, open_col, high_col, low_col, close_col]
    missing = [col for col in required if col not in intraday.columns]
    if missing:
        raise ValueError(f"Missing intraday column(s): {', '.join(missing)}")

    x = intraday[required].copy()
    x[datetime_col] = pd.to_datetime(x[datetime_col], errors="coerce")
    if isinstance(x[datetime_col].dtype, pd.DatetimeTZDtype):
        x[datetime_col] = x[datetime_col].dt.tz_localize(None)
    x = x.sort_values(datetime_col).dropna()
    x = x.rename(
        columns={datetime_col: "datetime", open_col: "open", high_col: "high", low_col: "low", close_col: "close"}
    )
    for col in ["open", "high", "low", "close"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna()
    if (x[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Intraday OHLC prices must be strictly positive.")

    x["date"] = x["datetime"].dt.normalize()
    x["log_close"] = np.log(x["close"])
    x["ret"] = x.groupby("date")["log_close"].diff()

    rows = []
    for date, full_day in x.groupby("date"):
        g = full_day.dropna(subset=["ret"])
        if len(g) < 2:
            continue
        open_px = float(full_day["open"].iloc[0])
        close_px = float(full_day["close"].iloc[-1])
        high_px = float(full_day["high"].max())
        low_px = float(full_day["low"].min())
        log_hl = math.log(high_px / low_px)
        log_co = math.log(close_px / open_px)
        rv = float((g["ret"] ** 2).sum())
        parkinson = (log_hl**2) / (4 * math.log(2))
        gk = max(0.0, 0.5 * (log_hl**2) - (2 * math.log(2) - 1) * (log_co**2))

        r = g["ret"].to_numpy()
        centered = r - r.mean()
        max_lag = min(kernel_lag, len(centered) - 1)
        kernel = float(np.dot(centered, centered))
        for lag in range(1, max_lag + 1):
            weight = 1 - lag / (max_lag + 1)
            kernel += 2 * weight * float(np.dot(centered[lag:], centered[:-lag]))

        rows.append(
            {
                "date": date,
                "Realized variance": max(rv, 1e-12),
                "Parkinson": max(parkinson, 1e-12),
                "Garman-Klass": max(gk, 1e-12),
                "Realized kernel": max(kernel, 1e-12),
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
            }
        )

    if not rows:
        raise ValueError("Not enough intraday bars per day.")
    out = pd.DataFrame(rows).set_index("date").sort_index()
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    log_open = np.log(out["open"])
    log_close = np.log(out["close"])
    overnight = (log_open - log_close.shift(1)).fillna(0)
    open_close = log_close - log_open
    out["Yang-Zhang"] = (overnight**2 + 0.34 * open_close**2 + 0.66 * out["Parkinson"]).clip(lower=1e-12)
    if estimator not in out.columns:
        raise ValueError(f"Unknown intraday estimator: {estimator}")
    out["realized_var"] = out[estimator].clip(lower=1e-12)
    out["realized_vol"] = np.sqrt(out["realized_var"] * TRADING_DAYS)
    return out
